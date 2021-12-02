import os
import sys
import time
import argparse

# from src.others.test_rouge_score import RougeScorer

PROBLEM = "ext"

## define paths 
PROJECT_DIR = os.getcwd() # current path
print(PROJECT_DIR)

DATA_DIR = f"{PROJECT_DIR}/datasets"
RAW_DATA_DIR = DATA_DIR + "/raw"
JSON_DATA_DIR = DATA_DIR + "/json"
BERT_DATA_DIR = DATA_DIR + "/bert"
LOG_DIR = f"{PROJECT_DIR}/{PROBLEM}/logs"
LOG_PREPO_FILE = LOG_DIR + "/preprocessing.log"

MODEL_DIR = f"{PROJECT_DIR}/{PROBLEM}/models"
RESULT_DIR = f"{PROJECT_DIR}/{PROBLEM}/results"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="ext", type=str, choices=["ext", "abs"])
    parser.add_argument(
        "--mode",
        default="test",
        type=str,
        choices=["install", "make_data", "train", "valid", "test"],
    )

    parser.add_argument("--n_cpus", default="2", type=str)
    parser.add_argument("--visible_gpus", default="0", type=str)
    parser.add_argument("--train_from", default=None, type=str)
    parser.add_argument("--use_model", default="kobert", type=str)
    parser.add_argument("--test_from", default=None, type=str)
    parser.add_argument("--make_gold", default="false", type=str)
    args = parser.parse_args()

    now = time.strftime('%m%d_%H%M')
    # now = "kykim_electra_0629_09"

    # python main.py -mode install
    if args.mode == "install":
        os.chdir(PROJECT_DIR)
        os.system("pip install -r requirements.txt")
        os.system("pip install -r requirements_prepro.txt")

    elif args.mode == "train":
        """
        파라미터별 설명은 trainer_ext 참고
        """
        os.chdir(PROJECT_DIR + "/src")

        do_str = (
            f"python train.py -task ext -mode train"
            + f" -model_path {MODEL_DIR} -bert_data_path {BERT_DATA_DIR}"
            + f" -visible_gpus {args.visible_gpus}"
        )

        if args.use_model == "kobert":
            use_model = "skt/kobert-base-v1"
        elif args.use_model == "distilkobert":
            use_model = "monologg/distilkobert"

        param = (
            " -ext_dropout 0.1 -max_pos 512 -batch_size 1000 -accum_count 2"
            + " -lr 2e-3 -warmup_steps 1200 "  # 전체가 21632개인데  평균 9.6문장이니 207667문장/ 5000 = 41.5
            + "  -train_steps 20000 -report_every 100"
            + f" -use_model {use_model}"
            + " -use_interval true"
        )
        do_str += param

        if args.train_from is None:
            os.system(f"mkdir {MODEL_DIR}/{now}")
            do_str += (
                f" -model_path {MODEL_DIR}/{now}"
                + f" -log_file {LOG_DIR}/train_{now}.log"
            )
        else:
            model_folder, model_name = args.train_from.rsplit("/", 1)
            do_str += (
                f" -train_from {MODEL_DIR}/{args.train_from}"
                + f" -model_path {MODEL_DIR}/{model_folder}"
                + f" -log_file {LOG_DIR}/train_{model_folder}.log"
            )

        print(do_str)
        os.system(do_str)

    # python main.py -mode test -test_from 1209_1236/model_step_7000.pt -visible_gpus 0
    elif args.mode == "test":
        os.chdir(PROJECT_DIR + "/src")

        model_folder, model_name = args.test_from.rsplit("/", 1)
        model_name = model_name.split("_", 1)[1].split(".")[0]

        do_str = "python train.py -task ext -mode test"
        
        if args.use_model == "kobert":
            use_model = "skt/kobert-base-v1"
        elif args.use_model == "distilkobert":
            use_model = "monologg/distilkobert"
        
        param = (f" -test_from {MODEL_DIR}/{args.test_from}" +
            f" -bert_data_path {PROJECT_DIR}/datasets/bert" +
            f" -result_path {RESULT_DIR}/result_{model_folder}" +
            f" -log_file {LOG_DIR}/test_{model_folder}.log" +
            " -test_batch_size 1 -batch_size 3000" +
            f" -sep_optim true -use_interval true -visible_gpus {args.visible_gpus}" +
            " -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50" +
            f" -report_rouge False -max_tgt_len 100 -make_gold {args.make_gold} -use_model {use_model}")

        do_str += param
        os.system(do_str)