import unittest
import sys
import os
from azure.ai.ml import MLClient, Input
from azure.ai.ml.constants import AssetTypes
from unittest.mock import patch, Mock, call
from azure.ai.ml.entities import PipelineJob
from azure.ai.ml.exceptions import ValidationException

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "app"))


@patch.dict(
    os.environ,
    {
        "SUBSCRIPTION_ID": "74cba509-39bd-4d44-a6e2-4c575956159c",
        "RESOURCE_GROUP_NAME": "ml-rg",
        "WORKSPACE_NAME": "pid-aml-workspace",
        "INPUT_IMAGE_DATA_PATH": "/images",
        "INPUT_LABEL_DATA_PATH": "/labels",
        "TRAINING_GPU_COMPUTE_NAME": "aml-compute-gpu",
        "PIPELINE_CPU_COMPUTE_NAME": "aml-compute-cpu",
        "MODEL_BASE_NAME": "pid-object-detection",
        "AZURE_TENANT_ID": "74cba509-39bd-4d44-a6e2-4c575956159c",
        "TRAINING_VALIDATION_DATA_SIZE": "0.3",
        "TRAINING_JOB_TIMEOUT_MINUTES": "180",
        "TRAINING_MAX_TRIALS": "5",
        "TRAINING_CONCURRENT_TRIALS": "5",
        "USE_STRATIFIED_SPLIT": "True",
        "IS_FAST_TRAINING": "FALSE",
        "AUTO_MODE": "True",
    },
    clear=True,
)
class TestPipeline(unittest.TestCase):
    @patch.object(MLClient, "jobs")
    def test_create_pipeline_job(self, mlclient_jobs_mock):
        from automl_pipeline.pipeline import create_pipeline_job

        pipeline_job_mock = Mock(PipelineJob)
        create_pipeline_job(pipeline_job_mock)

        assert mlclient_jobs_mock.validate.call_count == 1
        mlclient_jobs_mock.validate.assert_has_calls(
            [call(pipeline_job_mock, raise_on_failure=True)]
        )

        assert mlclient_jobs_mock.create_or_update.call_count == 1
        mlclient_jobs_mock.create_or_update.assert_has_calls(
            [call(pipeline_job_mock, experiment_name="pid-object-detection")]
        )

        assert mlclient_jobs_mock.stream.call_count == 1

    @patch.object(MLClient, "jobs")
    def test_create_pipeline_job_throws_error(self, mlclient_jobs_mock):
        from automl_pipeline.pipeline import create_pipeline_job

        mlclient_jobs_mock.validate.side_effect = ValidationException(
            "Validation failed on the following field -",
            "no_personal_data_message=no_personal_data_message",
        )

        pipeline_job_mock = Mock(PipelineJob)

        with self.assertRaises(ValidationException):
            create_pipeline_job(pipeline_job_mock)

        assert mlclient_jobs_mock.validate.call_count == 1
        mlclient_jobs_mock.validate.assert_has_calls(
            [call(pipeline_job_mock, raise_on_failure=True)]
        )

        assert mlclient_jobs_mock.create_or_update.call_count == 0
        assert mlclient_jobs_mock.stream.call_count == 0

    @patch("automl_pipeline.pipeline.object_detection_training")
    @patch("automl_pipeline.pipeline.ComponentsRepository")
    @patch("automl_pipeline.pipeline.DefaultAzureCredential")
    def test_pipeline_steps_setup(
        self,
        default_azure_credential_mock,
        components_repository_mock,
        object_detection_training_mock,
    ):
        from automl_pipeline.pipeline import pipeline_steps_setup

        data_aggregation_returner = Mock()
        data_aggregation_returner.outputs.output_path = (
            "azureml://datastore/pid_labelled_data"
        )
        components_repository_mock.data_aggregation.return_value = (
            data_aggregation_returner
        )

        data_splitter_returner = Mock()
        data_splitter_returner.outputs.train_output_path = (
            "azureml://datastore/train_data"
        )
        data_splitter_returner.outputs.val_output_path = "azureml://datastore/val_data"
        components_repository_mock.data_split.return_value = data_splitter_returner

        register_data_asset_returner = Mock()
        register_data_asset_returner.outputs.output_path = "azureml:dataset:1"
        components_repository_mock.register_data_asset.return_value = (
            register_data_asset_returner
        )

        object_detection_returner = Mock()
        object_detection_returner.outputs.best_model = "azureml://datastore/model_path"
        object_detection_training_mock.return_value = object_detection_returner

        register_returner = Mock()
        register_returner.outputs.model_metadata_path = (
            "azureml://datastore/model_metadata_path"
        )
        components_repository_mock.register.return_value = register_returner

        pipeline_steps_setup(
            input_image_data_path=Input(
                type=AssetTypes.URI_FOLDER,
                path="/images",
            ),
            input_label_data_path=Input(
                type=AssetTypes.URI_FOLDER,
                path="/labels",
            ),
            input_images_string_absolute_path="azureml://datastore/pid",
            mandatory_train_filenames=";",
            mandatory_val_filenames=";",
            model_base_name="pid-object-detection",
            build_source="build_source",
            build_id="build_number",
            data_store_name="workspaceblobstore",
            validation_data_size=0.2,
            stratified_split_n_fold=5,
        )

        assert components_repository_mock.data_aggregation.call_count == 1
        components_repository_mock.data_aggregation.assert_has_calls(
            [
                call(
                    input_image_data_path=Input(
                        type=AssetTypes.URI_FOLDER, path="/images"
                    ),
                    input_label_data_path=Input(
                        type=AssetTypes.URI_FOLDER, path="/labels"
                    ),
                    input_images_string_absolute_path="azureml://datastore/pid",
                    is_fast_training=False,
                )
            ]
        )

        assert components_repository_mock.data_split.call_count == 1
        components_repository_mock.data_split.assert_has_calls(
            [
                call(
                    input_data_path="azureml://datastore/pid_labelled_data",
                    mandatory_train_filenames=";",
                    mandatory_val_filenames=";",
                    stratified_split_n_fold=5,
                    use_stratified_split=True,
                )
            ]
        )

        assert components_repository_mock.register_data_asset.call_count == 1
        components_repository_mock.register_data_asset.assert_has_calls(
            [
                call(
                    train_input_path="azureml://datastore/train_data",
                    val_input_path="azureml://datastore/val_data",
                    data_store_name="workspaceblobstore",
                    use_stratified_split=True,
                )
            ]
        )

        assert object_detection_training_mock.call_count == 1
        object_detection_training_mock.assert_has_calls(
            [
                call(
                    train_data_path="azureml://datastore/train_data",
                    val_data_path="azureml://datastore/val_data",
                    training_job_timeout_minutes="180",
                    training_max_trials="5",
                    training_concurrent_trials="5",
                    training_gpu_compute_name="aml-compute-gpu",
                    is_fast_training=False,
                    use_stratified_split=True,
                    validation_data_size=0.2,
                    train_dataset=None,
                    val_dataset=None,
                    auto_mode=True,
                )
            ]
        )
        assert components_repository_mock.register.call_count == 1
        components_repository_mock.register.assert_has_calls(
            [
                call(
                    dataset_version="azureml:dataset:1",
                    model_input_path="azureml://datastore/model_path",
                    model_base_name="pid-object-detection",
                    build_source="build_source",
                    build_id="build_number",
                )
            ]
        )

        assert components_repository_mock.tag.call_count == 1
        components_repository_mock.tag.assert_has_calls(
            [
                call(
                    model_metadata_path="azureml://datastore/model_metadata_path",
                    model_base_name="pid-object-detection",
                )
            ]
        )
