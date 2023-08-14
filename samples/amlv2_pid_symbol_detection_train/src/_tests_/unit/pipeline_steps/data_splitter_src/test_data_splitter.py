import os
import sys
import unittest
import random
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
from app.pipeline_steps.data_splitter_src.data_splitter import (
    stratified_group_data_splitter,
    get_file_paths,
)


class TestDataSplitter(unittest.TestCase):
    def setUp(self):
        num_files = 50
        common_classes = ["A", "B", "C"]
        rare_class = "D"
        file_name_list = []
        y_list = []
        rare_class_file_idx = [10, 49]

        for file_idx in range(num_files):
            num_labels = random.randint(1, 5)
            for _ in range(num_labels):
                file_name_list.append(f"{file_idx}.jpg")
                y_list.append(random.choice(common_classes))
            if file_idx in rare_class_file_idx:
                file_name_list.append(f"{file_idx}.jpg")
                y_list.append(rare_class)

        self.sample_dataset = {
            "file_names": file_name_list,
            "y": y_list,
            "rare_classes": [rare_class],
        }

    def test_get_file_paths(self):
        all_filepaths = [
            "azureml://paths/0.jpg",
            "azureml://paths/0_upsampled_01.upsampled.jpg",
            "azureml://paths/1.jpg",
            "azureml://paths/1_upsampled_01.upsampled.jpg",
            "azureml://paths/1_upsampled_02.upsampled.jpg",
            "azureml://paths/2.jpg",
        ]

        search_file_1 = ["0.jpg", "2.jpg"]

        expected_1_1 = [
            "azureml://paths/0.jpg",
            "azureml://paths/0_upsampled_01.upsampled.jpg",
            "azureml://paths/2.jpg",
        ]

        expected_1_2 = [
            "azureml://paths/0.jpg",
            "azureml://paths/2.jpg",
        ]

        actual_1_1 = get_file_paths(
            search_filenames=search_file_1, all_filepaths=all_filepaths
        )

        search_file_1 = ["0.jpg", "2.jpg"]
        actual_1_2 = get_file_paths(
            search_filenames=search_file_1,
            all_filepaths=all_filepaths,
            include_upsample_files=False,
        )

        self.assertEqual(set(actual_1_1), set(expected_1_1))
        self.assertEqual(set(actual_1_2), set(expected_1_2))

        search_file_2 = ["4.jpg", "5.jpg"]
        expected_2 = all_filepaths = []

        actual_2_1 = get_file_paths(
            search_filenames=search_file_2, all_filepaths=all_filepaths
        )
        search_file_2 = ["4.jpg", "5.jpg"]
        actual_2_2 = get_file_paths(
            search_filenames=search_file_2,
            all_filepaths=all_filepaths,
            include_upsample_files=False,
        )

        self.assertEqual(set(actual_2_1), set(expected_2))
        self.assertEqual(set(actual_2_2), set(expected_2))

    def test_add_rare_to_val(self):
        file_names = self.sample_dataset["file_names"]
        y = self.sample_dataset["y"]

        train_files, val_files, rare_classes = stratified_group_data_splitter(
            file_names=file_names, labels=y, add_rare_to_val=True
        )

        # All training files are in either train and val
        assert set(self.sample_dataset["file_names"]).issubset(
            set(train_files).union(set(val_files))
        )
        # training and val dataset has no overlapped for those files without rare classes
        rare_idx = [i for i, x in enumerate(y) if x in rare_classes]
        rare_class_filenames = set(list(np.unique(np.array(file_names)[rare_idx])))
        assert (
            len(
                set(train_files)
                .difference(rare_class_filenames)
                .intersection(set(val_files).difference(rare_class_filenames))
            )
            == 0
        )
        # both train and val should have all the files with rare classes
        assert rare_class_filenames.issubset(set(train_files))
        assert rare_class_filenames.issubset(set(val_files))

    def test_rare_only_in_train(self):
        file_names = self.sample_dataset["file_names"]
        y = self.sample_dataset["y"]

        train_files, val_files, rare_classes = stratified_group_data_splitter(
            file_names=file_names, labels=y, add_rare_to_val=False
        )

        # All training files are in either train and val
        assert set(self.sample_dataset["file_names"]).issubset(
            set(train_files).union(set(val_files))
        )
        # there is no overlap in train and val
        assert len(set(train_files).intersection(set(val_files))) == 0
        # all of the rare file must in train
        rare_idx = [i for i, x in enumerate(y) if x in rare_classes]
        rare_class_filenames = set(list(np.unique(np.array(file_names)[rare_idx])))
        assert rare_class_filenames.issubset(set(train_files))
        # there is no rare class file in val
        len(rare_class_filenames.intersection(set(val_files))) == 0

    def test_mandatory_files(self):
        file_names = self.sample_dataset["file_names"]
        y = self.sample_dataset["y"]

        mandatory_train_filenames = list(set(file_names))[:3]
        mandatory_val_filenames = [list(set(file_names))[-3]]

        train_files, val_files, _ = stratified_group_data_splitter(
            file_names=file_names,
            labels=y,
            mandatory_train_filenames=mandatory_train_filenames,
            mandatory_val_filenames=mandatory_val_filenames,
        )
        # All training files are in either train and val
        assert set(self.sample_dataset["file_names"]).issubset(
            set(train_files).union(set(val_files))
        )
        # all file in mandatory_train_filenames in train, not in val
        assert set(mandatory_train_filenames).issubset(set(train_files))
        # all file in mandatory_val_filenames in val, not in train
        assert set(mandatory_val_filenames).issubset(set(val_files))

