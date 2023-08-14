import os
import sys
import unittest
import shutil
import json
from unittest.mock import patch, Mock, call
from mlflow.tracking.client import MlflowClient

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
from app.pipeline_steps.register_src.main import main


class TestMain(unittest.TestCase):
    _model_input_path = os.path.join(os.path.dirname(__file__), 'data', 'input')
    _model_base_name = 'test_model'
    _model_metadata_path = os.path.join(os.path.dirname(__file__), 'data', 'output')
    _expect_data_path = os.path.join(os.path.dirname(__file__), 'data', 'expect')


    def setUp(self):
        os.makedirs(self._model_metadata_path, exist_ok=True)


    def tearDown(self):
        shutil.rmtree(self._model_metadata_path)


    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.set_experiment')
    @patch('mlflow.register_model')
    @patch('mlflow.get_run')
    @patch.object(MlflowClient, 'set_model_version_tag')
    def test_main(self, set_model_version_tag_mock, mlflow_get_run, mlflow_register_model, mlflow_set_experiment, mlflow_set_tracking_uri):
        # arrange
        experiment_name = 'test_experiment'
        tracking_uri = 'test_tracking_uri'
        build_source = 'test_build_source'
        build_id = 'test_build_id'
        dataset_version = 'azureml:trainingset:1'

        mock_run = Mock()
        mock_run.data.metrics.get.return_value = 0.9
        mlflow_get_run.return_value = mock_run

        model_version_mock = Mock()
        model_version_mock.version = 1
        mlflow_register_model.return_value = model_version_mock

        # act
        main(
            self._model_input_path,
            self._model_base_name,
            self._model_metadata_path,
            experiment_name,
            tracking_uri,
            build_source,
            build_id,
            dataset_version)

        # assert
        expected_model_metadata = os.path.join(self._expect_data_path, 'model_metadata.json')
        actual_model_metadata = os.path.join(self._model_metadata_path, 'model_metadata.json')

        with open(expected_model_metadata, 'r') as expected, open(actual_model_metadata, 'r') as actual:
            expect_json = json.load(expected)
            actual_json = json.load(actual)

            assert expect_json == actual_json

        set_model_version_tag_mock.assert_has_calls([
            call(
                name=self._model_base_name,
                version=1,
                key="mean_average_precision",
                value=0.9),
            call(
                name=self._model_base_name,
                version=1,
                key="recall",
                value=0.9),
            call(
                name=self._model_base_name,
                version=1,
                key="precision",
                value=0.9),
            call(
                name=self._model_base_name,
                version=1,
                key="best_model",
                value='false'),
            call(
                name=self._model_base_name,
                version=1,
                key='build_source',
                value=build_source),
            call(
                name=self._model_base_name,
                version=1,
                key='build_id',
                value=build_id),
            call(
                name=self._model_base_name,
                version=1,
                key="dataset_version",
                value='azureml:trainingset:1')
        ])
