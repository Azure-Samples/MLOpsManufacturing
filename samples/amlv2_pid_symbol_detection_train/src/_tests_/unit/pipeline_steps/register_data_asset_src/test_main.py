import os
import sys
import unittest
from unittest.mock import call, patch, Mock

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
import app.pipeline_steps.register_data_asset_src.main

from app.pipeline_steps.register_data_asset_src.main import str2bool


class TestMain(unittest.TestCase):
    def test_get_args(self):
        # arrange
        train_input_path = "train_input_path"
        train_output_path = "train_output_path"
        output_path = "output_path"
        datastore_name = "datastore_name"
        workspace_name = "workspace_name"
        dataset_name = "dataset_name"
        dataset_version = "dataset_version"

        sys.argv = [
            "--train-input-path",
            "sample_artifacts/register_data_asset/input/train/annotations.jsonl",
            "--val-input-path",
            "sample_artifacts/register_data_asset/input/val/annotations.jsonl",
            "use_stratified_split",
            True,
            "--output-path",
            "sample_artifacts/register_data_asset/output/",
            "--data-store-name",
            "workspaceblobstore",
        ]

        with patch("argparse.ArgumentParser") as args_mock:
            # act
            app.pipeline_steps.register_data_asset_src.main.get_args()

            # assert
            args_mock.assert_has_calls(
                [
                    call().add_argument(
                        "--train-input-path",
                        dest="train_input_path",
                        type=str,
                        required=True,
                        help="Path to the folder containing the images to be used for training.",
                    ),
                    call().add_argument(
                        "--val-input-path",
                        dest="val_input_path",
                        type=str,
                        required=True,
                        help="Path to the folder containing the images to be used for validation.",
                    ),
                    call().add_argument(
                        "--use-stratified-split",
                        dest="use_stratified_split",
                        type=str2bool,
                        default=True,
                        help="Use stratified split or random split from automl.",
                    ),
                    call().add_argument(
                        "--output-path",
                        dest="output_path",
                        type=str,
                        required=True,
                        help="Path to the folder where the aggregated MLTable file will be saved.",
                    ),
                    call().add_argument(
                        "--data-store-name",
                        dest="data_store_name",
                        type=str,
                        required=True,
                        help="Data store name to register data asset.",
                    ),
                ]
            )

    def test_main(self):
        # arrange
        output_path = os.path.join(os.path.dirname(__file__), "data", "output.txt")
        train_data_path = os.path.join(os.path.dirname(__file__), "data", "train.jsonl")
        val_data_path = os.path.join(os.path.dirname(__file__), "data", "val.jsonl")
        dataset_version = (
            "train: azureml:training-data-set:1\tval: azureml:training-data-set:1\t"
        )

        with patch(
            "app.pipeline_steps.register_data_asset_src.main.get_uuid4"
        ) as uuid4_mock, patch(
            "app.pipeline_steps.register_data_asset_src.main.upload_directory"
        ) as upload_directory_mock, patch(
            "app.pipeline_steps.register_data_asset_src.main.mltable_from_json_lines_files"
        ) as mltable_from_json_lines_files_mock:
            workspace_mock = Mock()
            datastore_mock = Mock()
            workspace_mock.get_default_datastore.return_value = datastore_mock
            uuid4_mock.return_value = "CD471CB7-7996-47ED-BF7B-300E30BC39D7"

            mltable_mock = Mock()
            mltable_from_json_lines_files_mock.return_value = mltable_mock

            dataset = Mock()
            mltable_mock.register.return_value = dataset
            dataset.name = "training-data-set"
            dataset.version = 1
            use_stratified_split = True

            # act
            app.pipeline_steps.register_data_asset_src.main.main(
                train_data_path,
                val_data_path,
                use_stratified_split,
                output_path,
                datastore_mock,
                workspace_mock,
            )

            # assert
            assert upload_directory_mock.call_count == 2
            upload_directory_mock.assert_has_calls(
                [
                    call(
                        train_data_path,
                        datastore_mock,
                        "training-sets/CD471CB7-7996-47ED-BF7B-300E30BC39D7/train",
                    ),
                    call(
                        val_data_path,
                        datastore_mock,
                        "training-sets/CD471CB7-7996-47ED-BF7B-300E30BC39D7/val",
                    ),
                ]
            )

            mltable_from_json_lines_files_mock.call_count == 1
            mltable_from_json_lines_files_mock.assert_has_calls(
                [
                    call(
                        datastore_mock,
                        "training-sets/CD471CB7-7996-47ED-BF7B-300E30BC39D7/train/annotations.jsonl",
                    ),
                    call().register(
                        workspace_mock,
                        name="train-data-set-splitted",
                        description="train data set splitted",
                        create_new_version=True,
                    ),
                    call(
                        datastore_mock,
                        "training-sets/CD471CB7-7996-47ED-BF7B-300E30BC39D7/val/annotations.jsonl",
                    ),
                    call().register(
                        workspace_mock,
                        name="val-data-set-splitted",
                        description="val data set splitted",
                        create_new_version=True,
                    ),
                ]
            )

            with open(output_path, "r") as f:
                assert f.read() == dataset_version

            # clean up to delete the file
            os.remove(output_path)
