"""
data splitter
"""
from typing import List, Tuple, Optional
import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def get_file_paths(
    search_filenames: List[str],
    all_filepaths: List[str],
    include_upsample_files: bool = True,
    upsample_prefix="_upsampled_",
    upsample_ext=".upsampled",
) -> List[str]:
    """get the file paths containing in file names specified

    Args:
        search_filenames (List[str]): list of filenames to be searched. An example
         can be ["0.png", "1.png"]
        all_filepaths (List[str]): list of full path for all available files. An
         example can be ["azureml://paths/0.jpg", "azureml://paths/1.jpg",
         "azureml://paths/1_upsampled_01.upsampled.jpg", "azureml://paths/2.jpg]
        include_upsample_files (bool, optional): whether return the paths of files which
         upsampled from one of the search filename. Defaults to True.
        upsample_prefix (str, optional): part of file name to indicate a file is
         upsampeld from other files. Defaults to "_upsampled_".
        upsample_ext (str, optional): part of of file extension to indicate a file is
         upsampeld from other files. Defaults to ".upsampled".

    Returns:
        List[str]: the list of file paths containing in file names specified. An
         example is ["azureml://paths/0.jpg", "azureml://paths/1.jpg",
         "azureml://paths/1_upsampled_01.upsampled.jpg"]
    """
    selected_filepaths = [
        filepath
        for filepath in all_filepaths
        if any(
            Path(search_filename).name == Path(filepath).name
            for search_filename in search_filenames
        )
    ]
    if include_upsample_files:
        upsampled_prefix_list = [
            Path(filename).stem + upsample_prefix for filename in search_filenames
        ]
        upsampled_ext_list = [
            upsample_ext + Path(filename).suffix for filename in search_filenames
        ]
        selected_upsample_filepaths = [
            filepath
            for filepath in all_filepaths
            if any(
                Path(filepath).name.startswith(search_prefix)
                and Path(filepath).name.endswith(search_ext)
                for search_prefix, search_ext in zip(
                    upsampled_prefix_list, upsampled_ext_list
                )
            )
        ]
        selected_filepaths.extend(selected_upsample_filepaths)
    return selected_filepaths


def stratified_group_data_splitter(
    file_names: List[str],
    labels: List[str],
    n_splits: int = 5,
    add_rare_to_val: bool = True,
    mandatory_train_filenames: Optional[List[str]] = None,
    mandatory_val_filenames: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """Split the dataset into the trainning set and validation set,
     considering the balance of the datasets.
    To achieve this, we use the StratifiedGroupKFold method from
    sklearn.model_selection. This cross-validation object is a variation
    of StratifiedKFold attempts to return stratified folds with
    non-overlapping groups. The folds are made by
    preserving the percentage of samples for each class.
    Sptil ratio will be train:val=8:2

    Args:
        file_names (List[str]): file names for each bounding box annotation.
         file names don't need to be unique as one file can have
         multiple bounding boxes. each element in `file_names` represents one
         bounding box annotation data.
        y (List[str]): class labels for each bounding box. The order needs
         to align with `file_names`
        n_splits (int): number of splits applied wherein the first split will
         be used for validation set and the rest will go to training. 1/n_splits
         of all the available data is be used for validataion set and the rest are
         used as training set.
        add_rare_to_val (bool): whether add rare class data to validation set.
         Defaults to True
        mandatory_train_filenames (Optional[List[str]], optional): the files which
         must be included in train set, not in val set. Defaults to None.
        mandatory_val_filenames (Optional[List[str]], optional): the files which
         must be included in val set, not in train set. Defaults to None.

    Raises:
        ValueError: Raised when `file_names` length and `y` length don't match

    Returns:
        List[List[str], List[str]]: file names split into train,
        and validation sets.
        returned file names are all unique and there is no file name overlap
        across sets.
    """

    if len(file_names) != len(labels):
        raise ValueError(
            (
                "The length of `file_names` and `y` don't match."
                f"`file_names` got {len(file_names)} elements and `y` got {len(labels)}"
            )
        )

    if mandatory_train_filenames and mandatory_val_filenames:
        mandatory_overlapped = set(mandatory_train_filenames).intersection(
            set(mandatory_val_filenames)
        )
        if len(mandatory_overlapped) != 0:
            raise ValueError(f"overlapped mandatory file set at {mandatory_overlapped}")

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=33)
    # the first for val (1/n-split) and the rest for train
    splitter = iter(cv.split(file_names, y=labels, groups=file_names))
    _, val_idxs = next(splitter)
    train_mask = np.ones(len(file_names), dtype=bool)
    train_mask[val_idxs] = False
    file_name_array = np.array(file_names)
    y_array = np.array(labels)

    # set the splitting based on mandatory_{train|val}_filenames
    if mandatory_train_filenames:
        mandatory_train_filepaths = get_file_paths(
            search_filenames=mandatory_train_filenames,
            all_filepaths=list(set(file_names)),
        )
        print(f"mandatory_train_filepaths={mandatory_train_filepaths}")
        mandatory_train_idxs = np.in1d(
            file_name_array, np.array(list(mandatory_train_filepaths))
        ).nonzero()[0]
        train_mask[mandatory_train_idxs] = True
        val_idxs = np.setdiff1d(val_idxs, mandatory_train_idxs)

    if mandatory_val_filenames:
        mandatory_val_filepaths = get_file_paths(
            search_filenames=mandatory_val_filenames,
            all_filepaths=list(set(file_names)),
        )
        print(f"mandatory_val_filepaths={mandatory_val_filepaths}")
        mandatory_val_idxs = np.in1d(
            file_name_array, np.array(list(mandatory_val_filepaths))
        ).nonzero()[0]
        val_idxs = np.concatenate((val_idxs, mandatory_val_idxs))
        train_mask[mandatory_val_idxs] = False

    train_class = np.unique(y_array[train_mask])
    val_class = np.unique(y_array[val_idxs])
    rare_classes_in_train = set(train_class) - set(val_class)
    rare_classes_in_val = set(val_class) - set(train_class)
    rare_classes = (rare_classes_in_train).union(rare_classes_in_val)

    if len(rare_classes_in_val) != 0:
        warnings.warn(f"rare_classes_in_val = {rare_classes_in_val}")
    if len(rare_classes_in_train) != 0:
        warnings.warn(f"rare_classes_in_train = {rare_classes_in_train}")

    # cover corner cases when StratifiedGroupKFold is split more classes for
    # val compared to train. Rarely happens and need to correct if happened
    if rare_classes_in_val:
        warnings.warn(
            (
                "val has more classes then the training:"
                f"{rare_classes_in_val}."
                "shifting to training"
            )
        )
        val_rare_class_y_idxs = np.in1d(
            y_array, np.array(list(rare_classes_in_val))
        ).nonzero()[0]
        val_rare_class_filenames = np.unique(file_name_array[val_rare_class_y_idxs])
        val_rare_class_idxs = np.in1d(
            file_name_array, np.array(list(val_rare_class_filenames))
        ).nonzero()[0]
        val_idxs = np.setdiff1d(val_idxs, val_rare_class_idxs)
        train_mask[val_rare_class_idxs] = True

    if add_rare_to_val:
        if rare_classes:
            rare_class_y_idxs = np.in1d(
                y_array, np.array(list(rare_classes))
            ).nonzero()[0]
            rare_class_filenames = np.unique(file_name_array[rare_class_y_idxs])
            rare_class_idxs = np.in1d(
                file_name_array, np.array(list(rare_class_filenames))
            ).nonzero()[0]
            val_idxs = np.concatenate((val_idxs, rare_class_idxs))

    train_files = np.unique(file_name_array[train_mask]).tolist()
    val_files = np.unique(file_name_array[val_idxs]).tolist()

    return train_files, val_files, rare_classes
