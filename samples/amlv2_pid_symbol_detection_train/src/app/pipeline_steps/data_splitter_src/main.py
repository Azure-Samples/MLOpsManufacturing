import argparse
from pathlib import Path
from typing import List, Union, Optional, Any, Dict
import os
import json
import shutil

import jsonlines
import pandas as pd

from data_splitter import stratified_group_data_splitter


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def _get_class_count(jsonl_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """get class distribution from a list of jsonline from a dataset

    Args:
        jsonl_list (List[Dict[str, Any]]): list of jsonlines object of a
         dataset

    Returns:
        pd.DataFrame: dataframe describing the class distribution
    """

    class_list = []
    for jsonl in jsonl_list:
        for item in jsonl["label"]:
            class_name = item["label"]
            class_list.append(class_name)

    class_count = pd.DataFrame(class_list).value_counts().to_frame()
    class_count = class_count.rename(columns={0: "class_count"}).reset_index()
    class_count.rename(columns={0: "class_type"}, inplace=True)
    return class_count


def _create_ml_table_file(filename: str) -> str:
    """Create ML Table definition

    Args:
        filename (str): an specific annotation file to create the
         mltable configuration file; should with suffix of jsonl

    Returns:
        _type_: content of the config file in string
    """
    """"""

    return (
        "paths:\n"
        "  - file: ./{0}\n"
        "transformations:\n"
        "  - read_json_lines:\n"
        "        encoding: utf8\n"
        "        invalid_lines: error\n"
        "        include_path_column: false\n"
        "  - convert_column_types:\n"
        "      - columns: image_url\n"
        "        column_type: stream_info"
    ).format(filename)


def _save_ml_table_file(output_path, mltable_file_contents):
    with open(os.path.join(output_path, "MLTable"), "w") as f:
        f.write(mltable_file_contents)


def split_jsonl(
    jsonl_dir: Path,
    n_splits: int = 5,
    add_rare_to_val: bool = True,
    mandatory_train_filenames: Optional[List[str]] = None,
    mandatory_val_filenames: Optional[List[str]] = None,
) -> Union[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    take all the jsonls in a directory and split all the annotations in
    the directory into two stratified split of train and val

    Args:
        jsonl_dir (Path): location of the jsonl directory
        add_rare_to_val (bool, optional): whether add rare class data to
         validation set. Defaults to True
        n_splits (int): number of splits applied wherein the first split will
         be used for validation set and the rest will go to training. 1/n_splits
         of all the available data is be used for validataion set and the rest are
         used as training set.
        mandatory_train_filenames (Optional[List[str]], optional): the files which
         must be included in train set, not in val set. Defaults to None.
        mandatory_val_filenames (Optional[List[str]], optional): the files which
         must be included in val set, not in train set. Defaults to None.
    Returns:
        Union[List[Dict[str, Any]], List[Dict[str, Any]]]: the splitted jsonlines
    """

    jsonl_files = jsonl_dir.glob("*.jsonl")
    jsonl_list = []
    for file in jsonl_files:
        with jsonlines.open(file) as reader:
            for json_line in reader:
                jsonl_list.append(json_line)

    df_ann = pd.json_normalize(jsonl_list, record_path="label", meta="image_url")

    labels = df_ann["label"].values.tolist()
    filenames = df_ann["image_url"].values.tolist()

    splitted_filenames = stratified_group_data_splitter(
        labels=labels,
        file_names=filenames,
        n_splits=n_splits,
        add_rare_to_val=add_rare_to_val,
        mandatory_train_filenames=mandatory_train_filenames,
        mandatory_val_filenames=mandatory_val_filenames,
    )

    # getting the training and validation jsonl dictionaries
    train_jsonls, val_jsonls, _ = [
        [jsonl for jsonl in jsonl_list if jsonl["image_url"] in filenames]
        for filenames in splitted_filenames
    ]

    return train_jsonls, val_jsonls


def get_manditory_files(mandatory_filenames_str: str) -> List[str]:
    """get list of filenamse in based on mandatory_filenames_str

    Args:
        mandatory_filenames_str (str): list of filenames in string for
         files must in training set seperated by semicolon; example value is
         "32.jpg;33.jpg"

    Returns:
        List[str]: list of filenames in string format
    """
    mandatory_val_filenames = None
    if mandatory_filenames_str is not None:
        mandatory_val_filenames = [
            filename
            for filename in mandatory_filenames_str.strip().split(";")
            if filename != ""
        ]
        if len(mandatory_val_filenames) == 0:
            mandatory_val_filenames = None
    return mandatory_val_filenames


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-data-path",
        dest="input_data_path",
        type=str,
        required=True,
        help="Path to the folder splitted annoation for training and val.",
    )
    parser.add_argument(
        "--use-stratified-split",
        dest="use_stratified_split",
        type=str2bool,
        default=True,
        help="Use stratified split or random split from automl.",
    )
    parser.add_argument(
        "--stratified-split-n-fold",
        dest="stratified_split_n_fold",
        type=int,
        default=5,
        help="Use stratified split or random split from automl.",
    )
    parser.add_argument(
        "--mandatory-train-filenames",
        dest="mandatory_train_filenames",
        type=str,
        default=None,
        help="list of filenames must be included in training.",
    )
    parser.add_argument(
        "--mandatory-val-filenames",
        dest="mandatory_val_filenames",
        type=str,
        default=None,
        help="list of filenames must be included in val.",
    )
    parser.add_argument(
        "--train-output-path",
        dest="train_output_path",
        type=str,
        required=True,
        help="Path to the folder where the splitted training MLTable file will be saved.",
    )
    parser.add_argument(
        "--val-output-path",
        dest="val_output_path",
        type=str,
        required=True,
        help="Path to the folder where the splitted val MLTable file will be saved.",
    )

    args = parser.parse_args()
    return args


def main(
    input_data_path: str,
    train_output_path: str,
    val_output_path: str,
    mandatory_train_filenames_str: str,
    mandatory_val_filenames_str: str,
    use_stratified_split: bool = True,
    stratified_split_n_fold: int = 5,
):
    """split the data and save to train and val mltable

    Args:
        input_data_path (str): input jsonl path containing all data available
        train_output_path (str): splitted train dataset path
        val_output_path (str): splitted val dataset path
        mandatory_train_filenames_str (str): list of filenames in string for
         files must in training set seperated by semicolon; example value is
         "32.jpg;33.jpg"
        mandatory_val_filenames_str (str): llist of filenames in string for
         files must in val set seperated by semicolon; example value is
         "32.jpg;33.jpg"
    """
    print(f"use_stratified_split={use_stratified_split}")
    print(f"type(use_stratified_split)={type(use_stratified_split)}")

    jsonl_dir = Path(input_data_path)
    train_output_dir = Path(train_output_path)
    val_output_dir = Path(val_output_path)

    if not use_stratified_split:
        input_files = os.listdir(input_data_path)
        for fname in input_files:
            shutil.copy2(os.path.join(input_data_path, fname), train_output_path)
        val_non_json_path = Path(val_output_dir / "annotations.jsonl")
        with val_non_json_path.open("w") as dataset_f:
            dataset_f.write("")
        mltable_file_contents = _create_ml_table_file("annotations.jsonl")
        _save_ml_table_file(val_output_dir, mltable_file_contents)

    else:
        mandatory_train_filenames = get_manditory_files(mandatory_train_filenames_str)
        mandatory_val_filenames = get_manditory_files(mandatory_val_filenames_str)

        jsonl_train, jsonl_val = split_jsonl(
            jsonl_dir=jsonl_dir,
            mandatory_train_filenames=mandatory_train_filenames,
            mandatory_val_filenames=mandatory_val_filenames,
            n_splits=stratified_split_n_fold,
        )

        df_train_classes = _get_class_count(jsonl_train)
        df_val_classes = _get_class_count(jsonl_val)
        df_classes_overall = pd.merge(
            df_train_classes,
            df_val_classes,
            how="outer",
            on="class_type",
            suffixes=("_train", "_val"),
        )
        print(
            "class distribution cross train and val dataset"
            f" after splitting \n {df_classes_overall}"
        )

        for mltable_dir, jsonl_list in zip(
            [train_output_dir, val_output_dir], [jsonl_train, jsonl_val]
        ):
            mltable_dir.mkdir(exist_ok=True, parents=True)
            json_path = Path(mltable_dir / "annotations.jsonl")
            with json_path.open("w") as dataset_f:
                for jsonl in jsonl_list:
                    dataset_f.write(json.dumps(jsonl) + "\n")
            mltable_file_contents = _create_ml_table_file("annotations.jsonl")
            _save_ml_table_file(mltable_dir, mltable_file_contents)


if __name__ == "__main__":
    args = get_args()

    input_data_path = args.input_data_path
    train_output_path = args.train_output_path
    val_output_path = args.val_output_path
    mandatory_train_filenames_str = args.mandatory_train_filenames
    mandatory_val_filenames_str = args.mandatory_val_filenames
    use_stratified_split = args.use_stratified_split
    stratified_split_n_fold = args.stratified_split_n_fold

    main(
        input_data_path,
        train_output_path,
        val_output_path,
        mandatory_train_filenames_str,
        mandatory_val_filenames_str,
        use_stratified_split,
        stratified_split_n_fold,
    )
