import os
import unittest
from unittest.mock import patch, MagicMock

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
from app.automl_pipeline.deploy.scoring.online_endpoint_score import run, init


class TestRun(unittest.TestCase):
    def test_happy_path_get_request(self):
        # Arrange
        request = MagicMock()
        request.method = "GET"
        request.full_path = "/test"

        # Act
        response = run(request)

        # Assert
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.response, [b"/test"])


    @patch("app.automl_pipeline.deploy.scoring.online_endpoint_score.model")
    def test_happy_path_post_request(self, model_mock):
        # Arrange
        image_mock = MagicMock()
        image_mock.read.return_value = b"test"

        request = MagicMock()
        request.method = "POST"
        request.files = {"image": image_mock}

        model_predict_result = MagicMock()
        model_predict_result.to_json.return_value = '[{"test": "test"}]'
        model_mock.predict.return_value = model_predict_result

        # Act
        response = run(request)

        # Assert
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.response, [b'{"test": "test"}'])


    def test_when_no_image_in_request_then_returns_bad_request(self):
        # Arrange
        request = MagicMock()
        request.method = "POST"
        request.files = {}

        # Act
        response = run(request)

        # Assert
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.response, [b'{"message": "No image found in the request. Please send the request with an image that has a key of \\"image\\"."}'])


class TestInit(unittest.TestCase):
    _model_path = os.path.join(os.path.dirname(__file__), "input-data")

    @patch("app.automl_pipeline.deploy.scoring.online_endpoint_score.mlflow")
    def test_when_model_dir_does_not_exist_then_raises_exception(self, mlflow_mock):
        # Arrange
        base_path = self._model_path
        os.environ["AZUREML_MODEL_DIR"] = base_path
        mlflow_mock.pyfunc.load_model.return_value = "test"

        # Act
        init()
        del os.environ["AZUREML_MODEL_DIR"]

        # Assert
        expected_mlflow_path = os.path.join(self._model_path, "mlflow-model")
        mlflow_mock.pyfunc.load_model.assert_called_once_with(expected_mlflow_path)

    @patch("app.automl_pipeline.deploy.scoring.online_endpoint_score.mlflow")
    def test_when_model_dir_exists_then_loads_model(self, mlflow_mock):
        # Arrange
        os.environ["AZUREML_MODEL_DIR"] = os.path.join(self._model_path, "mlflow-model")
        mlflow_mock.pyfunc.load_model.return_value = "test"

        # Act
        init()
        del os.environ["AZUREML_MODEL_DIR"]

        # Assert
        expected_mlflow_path = os.path.join(self._model_path, "mlflow-model")
        mlflow_mock.pyfunc.load_model.assert_called_once_with(expected_mlflow_path)

    @patch("app.automl_pipeline.deploy.scoring.online_endpoint_score.mlflow")
    def test_when_model_fails_to_load_then_raises_exception(self, mlflow_mock):
        # Arrange
        os.environ["AZUREML_MODEL_DIR"] = self._model_path
        exception = Exception("test")
        mlflow_mock.pyfunc.load_model.side_effect = exception

        # Act
        with self.assertRaises(Exception) as error:
            init()

        del os.environ["AZUREML_MODEL_DIR"]

        # Assert
        self.assertEqual(str(error.exception), "test")
