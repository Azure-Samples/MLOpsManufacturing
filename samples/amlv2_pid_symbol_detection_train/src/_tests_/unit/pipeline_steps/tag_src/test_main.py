import sys
import os
import unittest
import shutil
from unittest.mock import patch, call
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version import ModelVersion
from mlflow.entities.model_registry.model_version_tag import ModelVersionTag
import json


sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
from app.pipeline_steps.tag_src.main import main


class TestMain(unittest.TestCase):
    _model_input_path = os.path.join(os.path.dirname(__file__), 'data', 'input')
    _model_base_name = 'test_model'
    _model_metadata_path = os.path.join(os.path.dirname(__file__), 'data', 'output')


    def setUp(self):
        os.makedirs(self._model_metadata_path, exist_ok=True)


    def tearDown(self):
        shutil.rmtree(self._model_metadata_path)


    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.set_experiment')
    @patch.object(MlflowClient, 'search_model_versions')
    @patch.object(MlflowClient, 'set_model_version_tag')
    def test_no_existing_models_and_best_model(self, set_model_version_tag_mock, search_model_versions_mock, set_experiment_mock, set_tracking_uri_mock):
        # arrange
        experiment_name = 'test_experiment'
        tracking_uri = 'test_tracking_uri'

        models = []
        search_model_versions_mock.return_value = models

        # act
        main(self._model_input_path,
             self._model_base_name,
             self._model_metadata_path,
             experiment_name,
             tracking_uri)

        # assert
        base_model_metadata = os.path.join(self._model_input_path, 'model_metadata.json')
        actual_model_metadata = os.path.join(self._model_metadata_path, 'best_model_metadata.json')

        with open(base_model_metadata, 'r') as base, open(actual_model_metadata, 'r') as actual:
            base_json = json.load(base)
            actual_json = json.load(actual)

            base_version = base_json['version']
            expected_result = { 'name': self._model_base_name, 'version': base_version }
            assert expected_result == actual_json


    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.set_experiment')
    @patch.object(MlflowClient, 'search_model_versions')
    @patch.object(MlflowClient, 'set_model_version_tag')
    def test_no_existing_best_model(self, set_model_version_tag_mock, search_model_versions_mock, set_experiment_mock, set_tracking_uri_mock):
        # arrange
        experiment_name = 'test_experiment'
        tracking_uri = 'test_tracking_uri'

        models = [
            ModelVersion(name=self._model_base_name, version=2, creation_timestamp=1),
            ModelVersion(name=self._model_base_name, version=3, creation_timestamp=2, tags=[ModelVersionTag('best_model', 'false')])
        ]
        search_model_versions_mock.return_value = models

        # act
        main(self._model_input_path,
             self._model_base_name,
             self._model_metadata_path,
             experiment_name,
             tracking_uri)

        # assert
        base_model_metadata = os.path.join(self._model_input_path, 'model_metadata.json')
        actual_model_metadata = os.path.join(self._model_metadata_path, 'best_model_metadata.json')

        with open(base_model_metadata, 'r') as base, open(actual_model_metadata, 'r') as actual:
            base_json = json.load(base)
            actual_json = json.load(actual)

            base_version = base_json['version']
            expected_result = { 'name': self._model_base_name, 'version': base_version }
            assert expected_result == actual_json

        set_tracking_uri_mock.assert_called_with(tracking_uri)
        set_experiment_mock.assert_called_with(experiment_name)
        search_model_versions_mock.assert_called_with(f"name = '{self._model_base_name}'")
        set_model_version_tag_mock.assert_called_with(
            name=self._model_base_name,
            version=1,
            key='best_model',
            value='true')


    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.set_experiment')
    @patch.object(MlflowClient, 'search_model_versions')
    def test_current_best_better_than_new_model(self, search_model_versions_mock, set_experiment_mock, set_tracking_uri_mock):
        # arrange
        experiment_name = 'test_experiment'
        tracking_uri = 'test_tracking_uri'

        models = [
            ModelVersion(name=self._model_base_name, version=2, creation_timestamp=1),
            ModelVersion(name=self._model_base_name, version=3, creation_timestamp=2, tags=[ModelVersionTag('best_model', 'true'), ModelVersionTag('mean_average_precision', '0.99')])
        ]
        search_model_versions_mock.return_value = models

        # act
        main(self._model_input_path,
             self._model_base_name,
             self._model_metadata_path,
             experiment_name,
             tracking_uri)

        # assert
        actual_model_metadata = os.path.join(self._model_metadata_path, 'best_model_metadata.json')

        with open(actual_model_metadata, 'r') as actual:
            actual_json = json.load(actual)

            expected_result = { 'name': self._model_base_name, 'version': 3 }
            assert expected_result == actual_json

        set_tracking_uri_mock.assert_called_with(tracking_uri)
        set_experiment_mock.assert_called_with(experiment_name)
        search_model_versions_mock.assert_called_with(f"name = '{self._model_base_name}'")


    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.set_experiment')
    @patch.object(MlflowClient, 'search_model_versions')
    @patch.object(MlflowClient, 'set_model_version_tag')
    def test_new_model_better_than_existing_best_model(self, set_model_version_tag_mock, search_model_versions_mock, set_experiment_mock, set_tracking_uri_mock):
        # arrange
        experiment_name = 'test_experiment'
        tracking_uri = 'test_tracking_uri'

        models = [
            ModelVersion(name=self._model_base_name, version=2, creation_timestamp=1),
            ModelVersion(name=self._model_base_name, version=3, creation_timestamp=2, tags=[ModelVersionTag('best_model', 'true'), ModelVersionTag('mean_average_precision', '0.1')])
        ]
        search_model_versions_mock.return_value = models

        # act
        main(self._model_input_path,
             self._model_base_name,
             self._model_metadata_path,
             experiment_name,
             tracking_uri)

        # assert
        base_model_metadata = os.path.join(self._model_input_path, 'model_metadata.json')
        actual_model_metadata = os.path.join(self._model_metadata_path, 'best_model_metadata.json')

        with open(base_model_metadata, 'r') as base, open(actual_model_metadata, 'r') as actual:
            base_json = json.load(base)
            actual_json = json.load(actual)

            base_version = base_json['version']
            expected_result = { 'name': self._model_base_name, 'version': base_version }
            assert expected_result == actual_json

        set_tracking_uri_mock.assert_called_with(tracking_uri)
        set_experiment_mock.assert_called_with(experiment_name)
        search_model_versions_mock.assert_called_with(f"name = '{self._model_base_name}'")
        set_model_version_tag_mock.assert_has_calls([
            call(
                name=self._model_base_name,
                version=1,
                key='best_model',
                value='true'),
            call(
                name=self._model_base_name,
                version=3,
                key='best_model',
                value='false')
        ])
