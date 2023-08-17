"""
Azure Auto ML object detection pipeline set up
"""
import datetime
from azure.ai.ml import MLClient, Input
from azure.identity import DefaultAzureCredential
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import PipelineJob
from automl_pipeline.mlops.image_object_detection_job import object_detection_training
from automl_pipeline.mlops.component_repository import ComponentsRepository
import automl_pipeline.constants as constants
from automl_pipeline.config import config

PIPELINE_DISPLAY_NAME = "pipeline" + datetime.datetime.now().strftime(
    "%Y-%m-%d-%H-%M-%S"
)


def pipeline_steps_setup(
    input_image_data_path,
    input_label_data_path,
    input_images_string_absolute_path,
    mandatory_train_filenames,
    mandatory_val_filenames,
    model_base_name,
    build_source,
    build_id,
    data_store_name,
    validation_data_size,
    stratified_split_n_fold,
) -> PipelineJob:
    """
    ML pipeline steps set up
    """
    if config.train_dataset is None:
        # Data aggregation step
        data_aggregation_step = ComponentsRepository.data_aggregation(
            input_image_data_path=input_image_data_path,
            input_label_data_path=input_label_data_path,
            input_images_string_absolute_path=input_images_string_absolute_path,
            is_fast_training=config.is_fast_training,
        )

        data_split_step = ComponentsRepository.data_split(
            input_data_path=data_aggregation_step.outputs.output_path,
            mandatory_train_filenames=mandatory_train_filenames,
            mandatory_val_filenames=mandatory_val_filenames,
            use_stratified_split=config.use_stratified_split,
            stratified_split_n_fold=stratified_split_n_fold,
        )

        register_data_asset_step = ComponentsRepository.register_data_asset(
            train_input_path=data_split_step.outputs.train_output_path,
            val_input_path=data_split_step.outputs.val_output_path,
            data_store_name=data_store_name,
            use_stratified_split=config.use_stratified_split,
        )

    object_training_detection_step = object_detection_training(
        train_data_path=""
        if config.train_dataset is not None
        else data_split_step.outputs.train_output_path,
        val_data_path=""
        if config.train_dataset is not None
        else data_split_step.outputs.val_output_path,
        training_job_timeout_minutes=config.training_job_timeout_minutes,
        training_max_trials=config.training_max_trials,
        training_concurrent_trials=config.training_concurrent_trials,
        training_gpu_compute_name=config.training_gpu_compute_name,
        is_fast_training=config.is_fast_training,
        use_stratified_split=config.use_stratified_split,
        validation_data_size=validation_data_size,
        train_dataset=config.train_dataset,
        val_dataset=config.val_dataset,
        auto_mode=config.auto_mode,
    )

    dataset_version_output = (
        Input(
            type=AssetTypes.URI_FILE,
            path=config.dataset_version_path,
            mode=InputOutputModes.RO_MOUNT,
        )
        if config.train_dataset is not None
        else register_data_asset_step.outputs.output_path
    )
    register_step = ComponentsRepository.register(
        model_input_path=object_training_detection_step.outputs.best_model,
        model_base_name=model_base_name,
        build_source=build_source,
        build_id=build_id,
        dataset_version=dataset_version_output,
    )

    _ = ComponentsRepository.tag(
        model_metadata_path=register_step.outputs.model_metadata_path,
        model_base_name=model_base_name,
    )


@pipeline(
    default_compute=config.pipeline_cpu_compute_name, display_name=PIPELINE_DISPLAY_NAME
)
def pipeline_steps_setup_proxy(
    input_image_data_path,
    input_label_data_path,
    input_images_string_absolute_path,
    mandatory_train_filenames,
    mandatory_val_filenames,
    model_base_name,
    build_source,
    build_id,
    data_store_name,
    validation_data_size,
    stratified_split_n_fold,
) -> PipelineJob:
    return pipeline_steps_setup(
        input_image_data_path,
        input_label_data_path,
        input_images_string_absolute_path,
        mandatory_train_filenames,
        mandatory_val_filenames,
        model_base_name,
        build_source,
        build_id,
        data_store_name,
        validation_data_size,
        stratified_split_n_fold,
    )


def create_pipeline_job(pipeline_job: PipelineJob):
    """
    Run the AML pipeline once the tasks are all defined
    """
    credential = DefaultAzureCredential()

    ml_client = MLClient(
        credential,
        config.subscription_id,
        config.resource_group_name,
        config.workspace_name,
    )

    ml_client.jobs.validate(pipeline_job, raise_on_failure=True)

    pipeline_job = ml_client.jobs.create_or_update(
        pipeline_job, experiment_name=constants.PIPELINE_JOB_EXPERIMENT_NAME
    )

    # wait for pipeline to finish
    ml_client.jobs.stream(pipeline_job.name)

    print(
        f"The pipeline {pipeline_job.name} has been submitted to {config.workspace_name}."
    )


def run_pipeline():
    pipeline_job = pipeline_steps_setup_proxy(
        input_image_data_path=Input(
            type=AssetTypes.URI_FOLDER,
            path=config.input_image_data_path,
        ),
        input_label_data_path=Input(
            type=AssetTypes.URI_FOLDER,
            path=config.input_label_data_path,
        ),
        input_images_string_absolute_path=config.input_image_data_path,
        mandatory_train_filenames=config.mandatory_train_filenames,
        mandatory_val_filenames=config.mandatory_val_filenames,
        model_base_name=config.model_base_name,
        build_source=config.build_source,
        build_id=config.build_id,
        data_store_name=config.data_store_name,
        validation_data_size=config.training_validation_data_size,
        stratified_split_n_fold=config.stratified_split_n_fold,
    )
    create_pipeline_job(pipeline_job)


if __name__ == "__main__":
    run_pipeline()
