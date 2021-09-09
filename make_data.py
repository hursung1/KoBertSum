import os
import sys
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
# from omegaconf import DictConfig
# import hydra
import argparse
import glob
from src.others.logging import logger

from src.prepro.preprocessor_kr import korean_sent_spliter, preprocess_kr

os.environ['MKL_THREADING_LAYER']='GNU'

def jsonl_to_df(params):
    from_dir = f"{os.getcwd()}/datasets/{params.dataset}"
    to_dir = f"{from_dir}/df"
    src_name, tgt_name, train_split, train_split_frac = (
        params.src_name,
        params.tgt_name,
        params.train_split, 
        params.train_split_frac,
    )

    # import data
    jsonl_file_paths = _get_file_paths(from_dir, ".jsonl")
    if not jsonl_file_paths:
        logger.error(f"There is no 'jsonl' files in {from_dir}")
        sys.exit("Stop")

    os.makedirs(to_dir, exist_ok=True)

    for jsonl_file_path in jsonl_file_paths:
        logger.info(f"Start 'jsonl_to_df' processing for {jsonl_file_path}...")
        # if "train" in jsonl_file_path:
        _jsonl_to_df(
            jsonl_file_path,
            to_dir,
            src_name,
            tgt_name=tgt_name,
            train_split=train_split,
            train_split_frac=train_split_frac,
        )


def _jsonl_to_df(
    jsonl_file_path, to_dir, src_name, tgt_name=None, train_split=True, train_split_frac=1.0
):
    def save_df(df, df_name, to_dir):
        df_path = f"{to_dir}/{df_name}.pickle"
        df.to_pickle(df_path)
        logger.info(f"Done! {df_path}({len(df)} rows) is exported")

    print(jsonl_file_path)
    subdata_group = _get_subdata_group(jsonl_file_path)
    with open(jsonl_file_path, "r") as json_file:
        json_list = list(json_file)

    jsons = []
    for json_str in json_list:
        line = json.loads(json_str)
        jsons.append(line)

    # Convert jsonl to df
    df = pd.DataFrame(jsons)

    if subdata_group in ["train", "valid"]:
        df = df[[src_name, tgt_name]]
        df = df.iloc[np.random.permutation(df.index)].reset_index(drop=True)

        if int(train_split_frac) != 1:  # train -> train / dev
            # random split
            train_df = df.sample(
                frac=train_split_frac, random_state=42
            )  # random state is a seed value
            valid_df = df.drop(train_df.index)
            if len(train_df) > 0:
                train_df.reset_index(inplace=True, drop=True)
                save_df(train_df, "train_df", to_dir)
            if len(valid_df) > 0:
                valid_df.reset_index(inplace=True, drop=True)
                save_df(valid_df, "valid_df", to_dir)
            
        else:  # just save df as train.df or valid.df
            if train_split:
                train_df = df
                save_df(train_df, to_dir)
            else:
                if subdata_group == 'train':
                    train_df = df
                    save_df(train_df, to_dir)
                elif subdata_group == 'valid':
                    valid_df = df
                    save_df(valid_df, to_dir)

    else:  # test
        test_df = df[[src_name]]
        save_df(test_df, "test_df", to_dir)


def df_to_bert(params):
    from_dir = f"{os.getcwd()}/datasets/{params.dataset}/df"
    temp_dir = f"{os.getcwd()}/datasets/{params.dataset}/json"
    to_dir = f"{os.getcwd()}/datasets/{params.dataset}/bert"
    log_file = f"{os.getcwd()}/datasets/{params.dataset}/data_prepro.log"
    src_name, tgt_name, tgt_type, train_split_frac = (
        params.src_name,
        params.tgt_name,
        params.tgt_type,
        params.train_split_frac,
    )
    n_cpus = params.n_cpus

    df_file_paths = _get_file_paths(from_dir, "pickle")
    if not df_file_paths:
        logger.error(f"There is no 'df' files in {from_dir}")
        sys.exit("Stop")

    # 동일한 파일명 존재하면 덮어쓰는게 아니라 ignore됨에 따라 폴더 내 삭제 후 만들어주기
    _make_or_initial_dir(temp_dir)
    _make_or_initial_dir(to_dir)
    # os.makedirs(to_dir, exist_ok=True)

    def _df_to_bert(df, subdata_group):
        # df to json file
        _df_to_json(
            df, src_name, tgt_name, tgt_type, temp_dir, subdata_group=subdata_group
        )

        # json to bert.pt files
        base_path = os.getcwd()
        _make_or_initial_dir(os.path.join(base_path, to_dir, subdata_group))
        os.system(
            f"python {os.path.join(base_path, 'src', 'preprocess.py')}"
            + f" -mode format_to_bert -dataset {subdata_group}"
            + f" -tgt_type {tgt_type}"
            + f" -raw_path {os.path.join(base_path, temp_dir)}"
            + f" -save_path {os.path.join(base_path, to_dir, subdata_group)}"
            + f" -log_file {os.path.join(base_path, log_file)}"
            + f" -lower -n_cpus {n_cpus}"
        )

    for df_file in df_file_paths:
        logger.info(f"Start 'df_to_bert' processing for {df_file}")
        df = pd.read_pickle(df_file)
        # print(df)
        subdata_group = _get_subdata_group(df_file)

        if (
            subdata_group == "train" and int(train_split_frac) != 1
        ):  # train -> train / dev
            # random split
            train_df = df.sample(
                frac=train_split_frac, random_state=42
            )  # random state is a seed value

            train_df = train_df[:5000]  ## 임시!!

            valid_df = df.drop(train_df.index)[:500]
            train_df.reset_index(inplace=True, drop=True)
            valid_df.reset_index(inplace=True, drop=True)

            _df_to_bert(train_df, subdata_group="train")
            _df_to_bert(valid_df, subdata_group="valid")

        else:
            _df_to_bert(df, subdata_group)


def _df_to_json(df, src_name, tgt_name, tgt_type, to_dir, subdata_group):
    NUM_DOCS_IN_ONE_FILE = 1000
    start_idx_list = list(range(0, len(df), NUM_DOCS_IN_ONE_FILE))

    for start_idx in tqdm(start_idx_list):
        end_idx = start_idx + NUM_DOCS_IN_ONE_FILE
        if end_idx > len(df):
            end_idx = len(df)  # -1로 하니 안됨...

        # 정렬을 위해 앞에 0 채워주기
        length = len(str(len(df)))
        start_idx_str = (length - len(str(start_idx))) * "0" + str(start_idx)
        end_idx_str = (length - len(str(end_idx - 1))) * "0" + str(end_idx - 1)

        file_name = f"{to_dir}/{subdata_group}.{start_idx_str}_{end_idx_str}.json"

        json_list = []
        for _, row in df.iloc[start_idx:end_idx].iterrows():
            src_sents = (
                row[src_name]
                if isinstance(row[src_name], list)
                else korean_sent_spliter(row[src_name])
            )
            original_sents_list = [preprocess_kr(sent).split() for sent in src_sents]

            summary_sents_list = []
            if subdata_group in ["train", "valid"]:
                if tgt_type == "idx_list":
                    summary_sents_list = row[tgt_name]
                else:
                    tgt_sents = (
                        row[tgt_name]
                        if isinstance(row[tgt_name], list)
                        else korean_sent_spliter(row[tgt_name])
                    )

                    summary_sents_list = [
                        preprocess_kr(sent).split() for sent in tgt_sents
                    ]
                # print("aaa", tgt_name, summary_sents_list)
            json_list.append({"src": original_sents_list, "tgt": summary_sents_list})

        json_string = json.dumps(json_list, indent=4, ensure_ascii=False)
        # print(json_string)
        with open(file_name, "w") as json_file:
            json_file.write(json_string)


def _get_file_paths(dir="./", suffix: str = ""):
    file_paths = []
    filename_pattern = f"*{suffix}" if suffix != "" else "*"

    file_paths = glob.glob(os.path.join(dir, filename_pattern))

    return file_paths


def _make_or_initial_dir(dir_path):
    """
    Make dir(When dir exists, remove it and make it)
    """
    if os.path.exists(dir_path):
        os.system(f"rm -rf {dir_path}/")
        logger.info(f"{dir_path} folder is removed")

    os.mkdir(dir_path)
    logger.info(f"{dir_path} folder is made")


def _get_subdata_group(path):
    filename = os.path.splitext(os.path.basename(path))[0]
    types = ["train", "valid", "test"]
    for type in types:
        if type in filename:
            print(f"---{type}, {filename}---")
            return type


# @ hydra.main(config_path='conf/make_data', config_name='config')
def main() -> None:
    parser = argparse.ArgumentParser(description="몰루")
    ### options as input
    parser.add_argument("--dataset", type=str, default="aihub", help="name of dataset")
    parser.add_argument("--train_file", type=str, default="train.jsonl", help="name of train dataset file")
    parser.add_argument("--valid_file", type=str, default="valid.jsonl", help="name of validation dataset file")
    parser.add_argument("--test_file", type=str, default="test.jsonl", help="name of test dataset file")
    parser.add_argument("--mode", type=str, default="jsonl_to_bert", help="mode of data deformation")
    parser.add_argument("--train_split", type=lambda s: s.lower() in ['true', '1'], default=False, help="split train set for validation data or not, true/false")
    parser.add_argument("--train_split_frac", type=float, default=1.0, help="proportion of train and validation set split")
    parser.add_argument("--n_cpus", type=int, default=1, help="# of cpus")
    
    ### 건들 필요 없는 부분
    parser.add_argument("--src_name", type=str, default="article_original", )
    parser.add_argument("--tgt_name", type=str, default="extractive", )
    parser.add_argument("--tgt_type", type=str, default="idx_list", )
    params = parser.parse_args()

    modes = ["jsonl_to_df", "jsonl_to_bert", "df_to_bert"]

    if params.mode not in modes:
        logger.error(f"Incorrect mode. Please choose one of {modes}")
        sys.exit("Stop")

    print(os.getcwd())

    # Convert raw data to df
    if params.mode in ["jsonl_to_df", "jsonl_to_bert"]:
        jsonl_to_df(params)

    # Make bert input file for train and valid from df file
    if params.mode in ["df_to_bert", "jsonl_to_bert"]:
        df_to_bert(params)


if __name__ == "__main__":
    main()
