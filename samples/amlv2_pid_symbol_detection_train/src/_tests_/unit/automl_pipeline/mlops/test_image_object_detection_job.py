import unittest
import sys
import os
from pathlib import Path
from azure.ai.ml import Output
from unittest.mock import patch, call

from azure.ai.ml.automl import SearchSpace
from azure.ai.ml.entities import Choice, Uniform, LogUniform

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "app"))
from automl_pipeline.mlops.image_object_detection_job import object_detection_training
from automl_pipeline.mlops.image_object_detection_job import _get_search_space


class TestImageObjectDetectionJob(unittest.TestCase):
    @patch("automl_pipeline.mlops.image_object_detection_job.image_object_detection")
    def test_object_detection_training(self, image_object_detection_mock):
        object_training_detection_step = object_detection_training(
            "azureml://datastore/any_datastore/paths/any_test",
            "azureml://datastore/any_datastore/paths/any_test",
            2,
            1,
            2,
            "hw-object-detection-nc24",
        )

        assert image_object_detection_mock.call_count == 1
        image_object_detection_mock.assert_has_calls(
            [
                call(
                    training_data="azureml://datastore/any_datastore/paths/any_test",
                    validation_data="azureml://datastore/any_datastore/paths/any_test",
                    target_column_name="label",
                    primary_metric="mean_average_precision",
                    outputs={"best_model": Output(type="mlflow_model")},
                    validation_data_size=None,
                ),
                call().set_limits(
                    timeout_minutes=2, max_trials=1, max_concurrent_trials=2
                ),
            ]
        )

        assert object_training_detection_step.set_limits.call_count == 1
        object_training_detection_step.set_limits.assert_has_calls(
            [call(timeout_minutes=2, max_trials=1, max_concurrent_trials=2)]
        )

        self.assertEqual(
            object_training_detection_step.compute, "hw-object-detection-nc24"
        )

    def test_get_search_space(self):
        config_path = (
            Path(os.path.dirname(__file__)) / "data/sample_config_search_space.yml"
        )
        actual_search_spaces = _get_search_space(config_path=config_path)
        self.assertEqual(len(actual_search_spaces), 1)

        actual_search_space = actual_search_spaces[0]

        self.assertEqual(actual_search_space.model_name, Choice(["yolov5"]))
        self.assertEqual(
            actual_search_space.learning_rate, Uniform(min_value=0.0001, max_value=0.01)
        )
        self.assertEqual(
            actual_search_space.image_size, LogUniform(min_value=640, max_value=800)
        )
