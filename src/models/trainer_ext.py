import os

import numpy as np
import torch
from tensorboardX import SummaryWriter
from operator import itemgetter

import distributed
from models.reporter_ext import ReportMgr, Statistics
from others.logging import logger
from others.utils import test_rouge, rouge_results_to_str
from copy import deepcopy


def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params


def build_trainer(args, device_id, model, optim):
    """
    Simplify `Trainer` creation based on user `opt`s*
    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """

    grad_accum_count = args.accum_count
    n_gpu = args.world_size

    if device_id >= 0:
        gpu_rank = int(args.gpu_ranks[device_id])
    else:
        gpu_rank = 0
        n_gpu = 0

    print("gpu_rank %d" % gpu_rank)

    tensorboard_log_dir = args.model_path

    writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")

    report_manager = ReportMgr(
        args.report_every, start_time=-1, tensorboard_writer=writer
    )

    trainer = Trainer(
        args, model, optim, grad_accum_count, n_gpu, gpu_rank, report_manager
    )

    # print(tr)
    if model:
        n_params = _tally_parameters(model)
        logger.info("* number of parameters: %d" % n_params)

    return trainer


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(
        self,
        args,
        model,
        optim,
        grad_accum_count=1,
        n_gpu=1,
        gpu_rank=1,
        report_manager=None,
    ):
        # Basic attributes.
        self.args = args
        self.valid_steps = args.valid_steps
        self.model = model
        self.optim = optim
        self.grad_accum_count = grad_accum_count
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.report_manager = report_manager
        self.stop_training = args.stop_training

        self.loss = torch.nn.BCELoss(reduction="none")
        assert grad_accum_count > 0
        # Set model in training mode.
        if model:
            self.model.train()

    def train(self, train_iter_fct, train_steps, valid_iter_fct=None, valid_steps=-1, k=1):
        """
        The main training loops.
        by iterating over training data (i.e. `train_iter_fct`)
        and running validation (i.e. iterating over `valid_iter_fct`

        Args:
            train_iter_fct(function): a function that returns the train
                iterator. e.g. something like
                train_iter_fct = lambda: generator(*args, **kwargs)
            valid_iter_fct(function): same as train_iter_fct, for valid data
            train_steps(int):
            valid_steps(int):

        Return:
            None
        """
        logger.info("Start training...")

        step = self.optim._step + 1
        true_batchs = []
        best_topk_models = []
        accum = 0
        normalization = 0
        stop_training_cnt = 0

        total_stats = Statistics() # total train stats
        report_stats = Statistics() # train stats for each steps
        valid_stats = Statistics() # total validation stats
        self._start_report_manager(start_time=total_stats.start_time)

        while stop_training_cnt < self.stop_training:
            self.model.train()
            train_iter = train_iter_fct()
            val_iter = valid_iter_fct()
            reduce_counter = 0
            for i, batch in enumerate(train_iter):
                if self.n_gpu == 0 or (i % self.n_gpu == self.gpu_rank):
                    true_batchs.append(batch)
                    normalization += batch.batch_size
                    accum += 1
                    if accum == self.grad_accum_count:
                        reduce_counter += 1
                        if self.n_gpu > 1:
                            normalization = sum(
                                distributed.all_gather_list(normalization)
                            )

                        self._gradient_accumulation(
                            true_batchs, normalization, total_stats, report_stats
                        )

                        report_stats = self._maybe_report_training(
                            step, train_steps, self.optim.learning_rate, report_stats
                        )

                        true_batchs = []
                        accum = 0
                        normalization = 0
                        if (
                            step % self.valid_steps == 0
                            and self.gpu_rank == 0
                        ): # validation for each self.save_checkpoint_steps
                            val_loss = self.validate(val_iter, valid_stats, step)
                            self.model.train()
                            if val_loss is not None:
                                print(f"Validation Loss: {val_loss}")
                                self._entry_topk(best_topk_models, val_loss, step, k, stop_training_cnt)
                                print(best_topk_models)

                        step += 1

        return total_stats, valid_stats

    def validate(self, valid_iter, valid_stats, step=0):
        """Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()
        total_loss = []

        with torch.no_grad():
            for batch in valid_iter:
                src = batch.src
                labels = batch.src_sent_labels
                segs = batch.segs
                clss = batch.clss
                mask = batch.mask_src
                mask_cls = batch.mask_cls

                sent_scores, mask = self.model(src, segs, clss, mask, mask_cls)
                loss = self.loss(sent_scores, labels.float())
                loss = (loss * mask.float()).sum()
                total_loss.append(loss)
                batch_stats = Statistics(float(loss.cpu().data.numpy()), len(labels))
                valid_stats.update(batch_stats)
            self._report_step(0, step, valid_stats=valid_stats)
            try:
                return sum(total_loss) / len(total_loss) # returns validation loss

            except ZeroDivisionError:
                return None


    def test(self, test_iter, step, cal_lead=False, cal_oracle=False):
        """Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """

        # Set model in validating mode.
        def _get_ngrams(n, text):
            ngram_set = set()
            text_length = len(text)
            max_index_ngram_start = text_length - n
            for i in range(max_index_ngram_start + 1):
                ngram_set.add(tuple(text[i : i + n]))
            return ngram_set

        def _block_tri(c, p):
            tri_c = _get_ngrams(3, c.split())
            for s in p:
                tri_s = _get_ngrams(3, s.split())
                if len(tri_c.intersection(tri_s)) > 0:
                    return True
            return False

        if not cal_lead and not cal_oracle:
            self.model.eval()
        stats = Statistics()

        can_path = f"{self.args.result_path}_step_{step}.candidate"
 
        with open(can_path, "w") as save_pred:
            with torch.no_grad():
                for batch in test_iter:
                    # import pdb; pdb.set_trace()
                    src = batch.src
                    labels = batch.src_sent_labels
                    segs = batch.segs
                    clss = batch.clss
                    mask = batch.mask_src
                    mask_cls = batch.mask_cls

                    gold = []
                    pred = []
                    pred_idx = []

                    if cal_lead:
                        selected_ids = [
                            list(range(batch.clss.size(1)))
                        ] * batch.batch_size
                    elif cal_oracle:
                        selected_ids = [
                            [
                                j
                                for j in range(batch.clss.size(1))
                                if labels[i][j] == 1
                            ]
                            for i in range(batch.batch_size)
                        ]
                    else:
                        sent_scores, mask = self.model(
                            src, segs, clss, mask, mask_cls
                        )

                        loss = self.loss(sent_scores, labels.float())
                        loss = (loss * mask.float()).sum()
                        batch_stats = Statistics(
                            float(loss.cpu().data.numpy()), len(labels)
                        )
                        stats.update(batch_stats)

                        sent_scores = sent_scores + mask.float()
                        sent_scores = sent_scores.cpu().data.numpy()
                        # print(sent_scores)
                        selected_ids = np.argsort(-sent_scores, 1)
                        # print(selected_ids)
                    # selected_ids = np.sort(selected_ids,1)
                    for i in range(len(selected_ids)):
                        _pred = []
                        _pred_idx = []
                        if len(batch.src_str[i]) == 0:
                            continue
                        for j in selected_ids[i][: len(batch.src_str[i])]:
                            if j >= len(batch.src_str[i]):
                                continue
                            candidate = batch.src_str[i][j].strip()
                            if self.args.block_trigram:
                                if not _block_tri(candidate, _pred):
                                    _pred.append(candidate)
                                    _pred_idx.append(j)
                            else:
                                _pred.append(candidate)
                                _pred_idx.append(j)

                            if (
                                (not cal_oracle)
                                and (not self.args.recall_eval)
                                and len(_pred) == 3
                            ):
                                break

                        # ????????? ???????????? ????????? sent??? json -> bert ????????? ????????? ????????? ??????????????? ??????
                        # min_src_ntokens_per_sent ??? ?????? ???????????? ??? ??? ????????????... ?????????.
                        # ????????? ???????????? 3?????? ????????? src?????? ?????? selected_ids[i]?????? ?????? ??????
                        # ????????? ????????? ?????? index??? ???????????????, ????????? ????????? ?????? ??????????????? ?????? index??? ?????? index ?????? ???????????????.... ?????? ???????????? ?????? ??????!!!!
                        if len(_pred_idx) < 3:
                            # print(_pred_idx)
                            # print('selected_ids: ', selected_ids)
                            # print('batch.src_str[i]: ', batch.src_str[i])
                            # print(f'selected_ids[{i}]: ', selected_ids[i])
                            if len(selected_ids[i]) >= 3:
                                # ????????? ???????????????. ???????????? ???????????????...!!!
                                # _pred = np.array(batch.src_str[i])[
                                #     selected_ids[i][:3]
                                # ]
                                _pred_idx = list(selected_ids[i][:3])
                                # print(_pred_idx)

                            else:
                                print(batch.src_str[i])
                                for naive_idx in range(3):
                                    if naive_idx not in _pred_idx:
                                        _pred_idx.append(naive_idx)
                                _pred_idx = _pred_idx[:3]

                        # print("labels", labels[i])  ??? ?????? 0??????... test??? ????????? ??????????!!!!
                        _pred = "<q>".join(_pred)
                        if self.args.recall_eval:
                            _pred = " ".join(
                                _pred.split()[: len(batch.tgt_str[i].split())]
                            )

                        pred.append(_pred)
                        pred_idx.append(_pred_idx)
                        # gold.append(batch.tgt_str[i])
                        
                    # for i in range(len(gold)):
                    #     sents = gold[i].split("<q>")
                    #     for sent in sents:
                    #         save_gold.write("<t>" + sent + "<\t>")
                        
                    for i in range(len(pred)):
                        # save_pred.write(pred[i].strip() + str(pred_idx[i]) + "\n")
                        sents = pred[i].split("<q>")
                        for sent in sents:
                            save_pred.write("<t>" + sent + "<\t>")

                    # save_gold.write("\n")
                    save_pred.write("\n")

        self._report_step(0, step, valid_stats=stats)

        return stats

    def _gradient_accumulation(
        self, true_batchs, normalization, total_stats, report_stats
    ):
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:
            if self.grad_accum_count == 1:
                self.model.zero_grad()

            src = batch.src
            labels = batch.src_sent_labels
            segs = batch.segs
            clss = batch.clss
            mask = batch.mask_src
            mask_cls = batch.mask_cls

            sent_scores, mask = self.model(src, segs, clss, mask, mask_cls)

            loss = self.loss(sent_scores, labels.float())
            loss = (loss * mask.float()).sum()
            (loss / loss.numel()).backward()
            # loss.div(float(normalization)).backward()

            batch_stats = Statistics(float(loss.cpu().data.numpy()), normalization)

            total_stats.update(batch_stats)
            report_stats.update(batch_stats)

            # 4. Update the parameters and statistics.
            if self.grad_accum_count == 1:
                # Multi GPU gradient gather
                if self.n_gpu > 1:
                    grads = [
                        p.grad.data
                        for p in self.model.parameters()
                        if p.requires_grad and p.grad is not None
                    ]
                    distributed.all_reduce_and_rescale_tensors(grads, float(1))
                self.optim.step()

        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.grad_accum_count > 1:
            if self.n_gpu > 1:
                grads = [
                    p.grad.data
                    for p in self.model.parameters()
                    if p.requires_grad and p.grad is not None
                ]
                distributed.all_reduce_and_rescale_tensors(grads, float(1))
            self.optim.step()

    def _save(self, step, del_step=None):
        """
        function to save model

        Parameters
        -----------
        step: when to save model
        del_step: delete model at step=del_step, default=None
        """
        real_model = self.model

        model_state_dict = real_model.state_dict()
        checkpoint = {
            "model": model_state_dict,
            "opt": self.args,
            "optim": self.optim,
        }
        checkpoint_path = os.path.join(self.args.model_path, "model_step_%d.pt" % step)
        logger.info("Saving checkpoint %s" % checkpoint_path)

        if del_step is not None:
            delete_path = os.path.join(self.args.model_path, f"model_step_{del_step}.pt")
            os.remove(delete_path)

        if not os.path.exists(checkpoint_path):
            torch.save(checkpoint, checkpoint_path)
            return checkpoint, checkpoint_path


    def _entry_topk(self, topk_models, val_loss, step, k, stop_training_cnt):
        """
        validation ????????? ?????? k??? ????????? ?????? ???????????? ???????????? ??????

        Parameters
        ------------
        topk_models: [(step1, val_loss1), (step2, val_loss2), ...], val_loss??? ?????? ?????????????????? ??????
        val_loss: ?????? step????????? validation loss
        step: ?????? step
        k: topk_model??? ????????? entry??? ???
        stop_training_cnt: counter, self.stop_training??? ???????????? ?????? ????????? ????????????.
        """
        del_step = None
        if len(topk_models) < k:
            topk_models.append((step, val_loss))
            
        else:
            if topk_models[k-1][1] > val_loss:
                del_step = topk_models[k-1][0]
                topk_models[k-1] = (step, val_loss)
                stop_training_cnt = 0
            else:
                stop_training_cnt += 1
        
        topk_models.sort(key=itemgetter(1))
        # save new model
        self._save(step, del_step)


    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases

        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.n_gpu > 1:
            return Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate, report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats, multigpu=self.n_gpu > 1
            )

    def _report_step(self, learning_rate, step, train_stats=None, valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats, valid_stats=valid_stats
            )

    def _maybe_save(self, step):
        """
        Save the model if a model saver is set
        """
        if self.model_saver is not None:
            self.model_saver.maybe_save(step)
