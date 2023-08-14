"""
Environment variables for the different modules
"""

import os
from dotenv import load_dotenv

load_dotenv()

DEFAULT_BUILD_ID = "local"
DEFAULT_BUILD_SOURCE = "local"


class Config(object):
    """
    Environment variables for different modules
    """

    _subscription_id = None
    _resource_group_name = None
    _workspace_name = None

    _input_image_data_path = None
    _input_label_data_path = None

    _stratified_split_n_fold = None
    _use_stratified_split = None
    _mandatory_train_filenames = None
    _mandatory_val_filenames = None

    _training_gpu_compute_name = None
    _training_validation_data_size = None
    _training_job_timeout_minutes = None
    _training_max_trials = None
    _training_concurrent_trials = None
    _train_dataset = None
    _val_dataset = None
    _dataset_version_path = None

    _pipeline_cpu_compute_name = None
    _model_base_name = None

    _build_number = str()
    _build_source = str()

    def __init__(self):
        self._subscription_id = os.getenv("SUBSCRIPTION_ID")
        self._resource_group_name = os.getenv("RESOURCE_GROUP_NAME")
        self._workspace_name = os.getenv("WORKSPACE_NAME")

        # Data aggregation step
        self._input_image_data_path = os.getenv("INPUT_IMAGE_DATA_PATH")
        self._input_label_data_path = os.getenv("INPUT_LABEL_DATA_PATH")

        # data split step
        self._stratified_split_n_fold = os.getenv("STRATIFIED_SPLIT_N_FOLD", default=5)
        self._use_stratified_split = str2bool(
            os.getenv("USE_STRATIFIED_SPLIT", default=True)
        )
        self._mandatory_train_filenames = os.getenv(
            "MANDATORY_TRAIN_FILENAMES", default=";"
        )
        self._mandatory_val_filenames = os.getenv(
            "MANDATORY_VAL_FILENAMES", default=";"
        )

        # model training variables
        self._training_gpu_compute_name = os.getenv("TRAINING_GPU_COMPUTE_NAME")
        self._training_validation_data_size = os.getenv(
            "TRAINING_VALIDATION_DATA_SIZE", default=0.2
        )
        self._training_job_timeout_minutes = os.getenv(
            "TRAINING_JOB_TIMEOUT_MINUTES", default=120
        )
        self._training_max_trials = os.getenv("TRAINING_MAX_TRIALS", default=10)
        self._training_concurrent_trials = os.getenv(
            "TRAINING_CONCURRENT_TRIALS", default=4
        )
        self._train_dataset = os.getenv("TRAIN_DATASET")
        self._val_dataset = os.getenv("VAL_DATASET")
        self._dataset_version_path = os.getenv("DATASET_VERSION_PATH")

        # build variables
        self._build_id = os.getenv("BUILD_ID", default=DEFAULT_BUILD_ID)
        self._build_source = os.getenv("BUILD_SOURCE", default=DEFAULT_BUILD_SOURCE)

        self._pipeline_cpu_compute_name = os.getenv("PIPELINE_CPU_COMPUTE_NAME")
        self._model_base_name = os.getenv("MODEL_BASE_NAME")

        _azure_tenant_id = os.getenv("AZURE_TENANT_ID")

        self._is_fast_training = str2bool(os.getenv("IS_FAST_TRAINING", default=False))
        self._data_store_name = os.getenv(
            "DATA_STORE_NAME", default="workspaceblobstore"
        )
        self._auto_mode = str2bool(os.getenv("AUTO_MODE", default=True))

        if _azure_tenant_id is None:
            raise ValueError("Azure Tenant ID not set")

        if self._subscription_id is None:
            raise ValueError("Subscription ID not set")

        if self._resource_group_name is None:
            raise ValueError("Resource Group Name not set")

        if self._workspace_name is None:
            raise ValueError("Workspace Name not set")

        if self._pipeline_cpu_compute_name is None:
            raise ValueError("Pipeline Compute Name not set")

        if self._input_image_data_path is None:
            raise ValueError("Input Image Data Path not set")

        if self._input_label_data_path is None:
            raise ValueError("Input Label Data Path not set")

        if self._training_gpu_compute_name is None:
            raise ValueError("Training Compute Name not set")

        if self._training_max_trials is None:
            raise ValueError("Training Max Trials not set")

        if self._training_concurrent_trials is None:
            raise ValueError("Training Concurrent Trials not set")

        if self._training_job_timeout_minutes is None:
            raise ValueError("Training Job Timeout Minutes not set")

        if self._training_validation_data_size is None:
            raise ValueError("Training Validation Data Size not set")

        if self._model_base_name is None:
            raise ValueError("Model base name not set")

        if self._train_dataset is not None and self._dataset_version_path is None:
            raise ValueError("Dataset version not set while exiting datset is True")

        if not self._auto_mode and self._is_fast_training:
            raise ValueError(
                "Fast training should be turned off for manual search. "
                "Please only enable one from AUTO_MODE and IS_FAST_TRAINING."
            )

    @property
    def subscription_id(self) -> str:
        """
        Azure resources subscription id

        Returns:
            str: subscription id
        """
        return self._subscription_id

    @property
    def resource_group_name(self) -> str:
        """
        Azure resources resource group name

        Returns:
            str: resource group name
        """
        return self._resource_group_name

    @property
    def workspace_name(self) -> str:
        """
        Azure resources workspace name

        Returns:
            str: workspace name
        """
        return self._workspace_name

    @property
    def input_image_data_path(self) -> str:
        return self._input_image_data_path

    @property
    def input_label_data_path(self) -> str:
        return self._input_label_data_path

    @property
    def use_stratified_split(self) -> int:
        return self._use_stratified_split

    @property
    def stratified_split_n_fold(self) -> int:
        return self._stratified_split_n_fold

    @property
    def mandatory_train_filenames(self) -> str:
        return self._mandatory_train_filenames

    @property
    def mandatory_val_filenames(self) -> str:
        return self._mandatory_val_filenames

    @property
    def training_gpu_compute_name(self) -> str:
        """
        object detection model training job compute name

        Returns:
            str: compute name for the training job
        """
        return self._training_gpu_compute_name

    @property
    def training_validation_data_size(self) -> float:
        """
        object detection model training job validation data size argument

        Returns:
            float: validation data size for the training and validation data size
        """
        return self._training_validation_data_size

    @property
    def training_job_timeout_minutes(self) -> int:
        """
        object detection model training job timeout argument

        Returns:
            int: timeout in minutes for the compute configured
        """
        return self._training_job_timeout_minutes

    @property
    def training_max_trials(self) -> int:
        """
        object detection model training job max trials argument

        Returns:
            int: max trials for the compute configured
        """
        return self._training_max_trials

    @property
    def training_concurrent_trials(self) -> int:
        """
        object detection model training job concurrent trials argument

        Returns:
            int: allowed concurrent trials for the compute configured
        """
        return self._training_concurrent_trials

    @property
    def pipeline_cpu_compute_name(self) -> str:
        """
        object detection model pipeline compute name

        Returns:
            str: compute name for the pipeline
        """
        return self._pipeline_cpu_compute_name

    @property
    def model_base_name(self) -> str:
        """
        object detection model registration name

        Returns:
            str: model base name for the pipeline
        """
        return self._model_base_name

    @property
    def is_fast_training(self) -> bool:
        """
        control the number of epochs, trials and images for the training job

        Returns:
            bool: fast training enabled or not
        """
        return self._is_fast_training

    @property
    def auto_mode(self) -> bool:
        """
        turn on paremeter tuning auto_mode for object detection trainng job

        Returns:
            bool: turn on paremeter tuning auto_mode for object detection trainng job
        """
        return self._auto_mode

    @property
    def train_dataset(self) -> str:
        """
        exisiting training dataset uesed

        Returns:
            training dataset uesed
        """
        return self._train_dataset

    @property
    def val_dataset(self) -> str:
        """
        exisiting validation dataset uesed

        Returns:
            bool: exisiting validation dataset uesed
        """
        return self._val_dataset

    @property
    def dataset_version_path(self) -> str:
        """
        existing dataset version file path used if is_fix_dataset is True

        Returns:
            bool: existing dataset version file path used if is_fix_dataset is True
        """
        return self._dataset_version_path

    @property
    def data_store_name(self) -> str:
        """
        data store name for registering training data assets

        Returns:
            str: data store name
        """
        return self._data_store_name

    @property
    def build_id(self) -> str:
        """
        The pipeline build id ("local" if a local run else "pipeline")

        Returns:
            str: The pipeline build id
        """
        return self._build_id

    @property
    def build_source(self) -> str:
        """
        The source that built the pipeline

        Returns:
            str: The pipeline build source
        """
        return self._build_source


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError("Boolean value expected.")


config = Config()
