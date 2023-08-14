"""
Code for training the object detection model
"""
from typing import List
import os
from warnings import warn
from pathlib import Path
import yaml

from azure.ai.ml import Output
from azure.ai.ml.automl import image_object_detection
from azure.ai.ml.automl import SearchSpace
from azure.ai.ml.entities import Choice, Uniform, LogUniform

import automl_pipeline.constants as constants
from automl_pipeline.mlops.object_detection_search_space import (
    object_detection_training_parameters,
)


def _get_search_space(config_path: Path) -> List[SearchSpace]:
    """get the search space from config_search_space.yml

    Raises:
        ValueError: raise if yaml is not parable or format is not correct

    Returns:
        List[SearchSpace]: list of SearchSpace for the training job
    """
    with open(config_path, "r") as stream:
        try:
            config_search_space_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return []

    if "search_space" in config_search_space_dict:
        config_search_space = config_search_space_dict["search_space"]
    else:
        raise ValueError("missing configuration key: search_space")

    search_spaces = []
    for config_sub_search_space in config_search_space:
        dict_sub_search_space = {}
        for parameter, value in config_sub_search_space.items():
            if parameter in object_detection_training_parameters:
                if value["type"] == "choice":
                    dict_sub_search_space[parameter] = Choice(value["values"])
                elif value["type"] == "uniform":
                    dict_sub_search_space[parameter] = Uniform(
                        min_value=value["min_value"], max_value=value["max_value"]
                    )
                elif value["type"] == "loguniform":
                    dict_sub_search_space[parameter] = LogUniform(
                        min_value=value["min_value"], max_value=value["max_value"]
                    )
                else:
                    unkown_type = value["type"]
                    warn(f"Unknown search space type named {unkown_type}")
            else:
                warn(f"Unknown training parameter: {parameter}")

        sub_search_space = SearchSpace(**dict_sub_search_space)
        search_spaces.append(sub_search_space)

    return search_spaces


def str2bool(v: str) -> bool:
    """convert a string to bool type

    Args:
        v (str): string value to convert

    Raises:
        ValueError: raises if the string is not in right form

    Returns:
        bool: boolean converted
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError(f"the string {v} is not of convertable form for boolean")


def object_detection_training(
    train_data_path: str,
    val_data_path: str,
    training_job_timeout_minutes: int,
    training_max_trials: int,
    training_concurrent_trials: int,
    training_gpu_compute_name: str,
    is_fast_training: bool = False,
    use_stratified_split: bool = True,
    validation_data_size: float = 0.2,
    train_dataset: str = None,
    val_dataset: str = None,
    auto_mode: bool = True,
):
    """
    Training the model using the input data for object detection
    """
    is_fast_training = str2bool(is_fast_training)
    if train_dataset is not None:
        image_object_detection_node = image_object_detection(
            training_data=train_dataset,
            validation_data=val_dataset,
            validation_data_size=validation_data_size,
            target_column_name=constants.TRAINING_TARGET_COLUMN_NAME,
            primary_metric=constants.TRAINING_PRIMARY_METRIC,
            outputs={"best_model": Output(type="mlflow_model")},
        )

    else:
        use_stratified_split = str2bool(use_stratified_split)
        validation_data = val_data_path if use_stratified_split else None
        train_validation_data_size = (
            None if use_stratified_split else validation_data_size
        )
        image_object_detection_node = image_object_detection(
            training_data=train_data_path,
            validation_data=validation_data,
            validation_data_size=train_validation_data_size,
            target_column_name=constants.TRAINING_TARGET_COLUMN_NAME,
            primary_metric=constants.TRAINING_PRIMARY_METRIC,
            outputs={"best_model": Output(type="mlflow_model")},
        )

    image_object_detection_node.set_limits(
        timeout_minutes=training_job_timeout_minutes,
        max_trials=training_max_trials,
        max_concurrent_trials=training_concurrent_trials,
    )

    image_object_detection_node.compute = training_gpu_compute_name

    if auto_mode and is_fast_training:
        image_object_detection_node.set_training_parameters(
            early_stopping=True,
            evaluation_frequency=1,
            random_seed=3,
            box_detections_per_image=1,
            box_score_threshold=0.001,
        )

        # Define search space
        image_object_detection_node.extend_search_space(
            [
                SearchSpace(
                    learning_rate=Choice([0.03]),
                    number_of_epochs=Choice([1]),
                    model_name=Choice(["yolov5"]),
                    model_size=Choice(["small"]),
                )
            ]
        )
    elif not auto_mode:
        config_path = Path(os.path.dirname(__file__)) / "config_search_space.yml"
        if config_path.exists():
            search_spaces = _get_search_space(config_path=config_path)
            if search_spaces is not None and len(search_spaces) > 0:
                image_object_detection_node.extend_search_space(search_spaces)
        else:
            raise RuntimeError(f"there is no configuration file at {config_path}")

    return image_object_detection_node
