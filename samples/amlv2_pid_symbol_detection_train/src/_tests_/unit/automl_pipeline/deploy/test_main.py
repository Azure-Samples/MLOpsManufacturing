import os
import unittest
from unittest.mock import patch, Mock, ANY, call
from azure.ai.ml.entities import Model
from azure.core.exceptions import ResourceNotFoundError
from azure.ai.ml.exceptions import LocalEndpointNotFoundError

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "app"))
from automl_pipeline.deploy.main import get_args, create_online_endpoint


class TestGetArgs(unittest.TestCase):
    @patch(
            "sys.argv",
            [
               "main.py",
               "--subscription-id",
               "00000000-0000-0000-0000-000000000000",
               "--resource-group-name",
               "hw-pid-dev-rg",
               "--workspace-name",
               "hw-pid-dev-mlw",
               "--environment-image",
               "test_image"
            ]
    )
    def test_only_required_args(self):
        # arrange

        # act
        args = get_args()

        # assert
        self.assertEqual(args.subscription_id, "00000000-0000-0000-0000-000000000000")
        self.assertEqual(args.resource_group_name, "hw-pid-dev-rg")
        self.assertEqual(args.workspace_name, "hw-pid-dev-mlw")
        self.assertEqual(args.is_local_deployment, False)
        self.assertEqual(args.endpoint_name, "pid-symbol-detection-endpoint")
        self.assertEqual(args.endpoint_description, "The PID symbol detection endpoint")
        self.assertEqual(args.model_name, None)
        self.assertEqual(args.model_version, None)
        self.assertEqual(args.environment_image, "test_image")
        self.assertEqual(args.compute_instance_type, "Standard_DS3_v2")
        self.assertEqual(args.compute_instance_count, 1)

    @patch(
        "sys.argv",
        [
            "main.py",
            "--subscription-id",
            "00000000-0000-0000-0000-000000000000",
            "--resource-group-name",
            "hw-pid-dev-rg",
            "--workspace-name",
            "hw-pid-dev-mlw",
            "--is-local-deployment",
            "--endpoint-name",
            "test_endpoint",
            "--endpoint-description",
            "test_description",
            "--model-name",
            "test_model",
            "--model-version",
            "test_version",
            "--environment-image",
            "test_image",
            "--compute-instance-type",
            "test_type",
            "--compute-instance-count",
            "2",
        ],
    )
    def test_all_args(self):
        # arrange

        # act
        args = get_args()

        # assert
        self.assertEqual(args.subscription_id, "00000000-0000-0000-0000-000000000000")
        self.assertEqual(args.resource_group_name, "hw-pid-dev-rg")
        self.assertEqual(args.workspace_name, "hw-pid-dev-mlw")
        self.assertEqual(args.is_local_deployment, True)
        self.assertEqual(args.endpoint_name, "test_endpoint")
        self.assertEqual(args.endpoint_description, "test_description")
        self.assertEqual(args.model_name, "test_model")
        self.assertEqual(args.model_version, "test_version")
        self.assertEqual(args.environment_image, "test_image")
        self.assertEqual(args.compute_instance_type, "test_type")
        self.assertEqual(args.compute_instance_count, 2)


class TestCreateOnlineEndpoint(unittest.TestCase):
    def test_when_local_deployment(self):
        # arrange
        ml_client = Mock()
        is_local_deployment = True
        endpoint_name = "test_endpoint"
        endpoint_description = "test_description"
        model_name = "test_model"
        model_version = "test_version"
        environment_image = "test_image"
        deployment_name = "test_deployment"
        compute_instance_type = "test_type"
        compute_instance_count = 2
        scoring_uri = "test_scoring_uri"
        request_in_ms = 60000

        mock_online_endpoint = Mock()
        mock_online_endpoint.scoring_uri = scoring_uri

        ml_client.online_endpoints.begin_create_or_update.return_value = None
        ml_client.online_deployments.begin_create_or_update.return_value = None
        ml_client.online_endpoints.get.side_effect = [
            LocalEndpointNotFoundError(endpoint_name),
            mock_online_endpoint,
        ]

        # act
        actual_online_deployment, actual_online_endpoint = create_online_endpoint(
            ml_client,
            is_local_deployment,
            endpoint_name,
            endpoint_description,
            model_name,
            model_version,
            environment_image,
            deployment_name,
            compute_instance_type,
            compute_instance_count,
            request_in_ms,
        )

        # assert
        self.assertEqual(actual_online_deployment.name, deployment_name)
        self.assertEqual(actual_online_endpoint.scoring_uri, scoring_uri)

        ml_client.online_endpoints.begin_create_or_update.assert_called_once_with(
            ANY, local=True
        )
        ml_client.online_deployments.begin_create_or_update.assert_called_once_with(
            deployment=ANY, local=True
        )
        ml_client.online_endpoints.get.assert_has_calls(
            [call(name=endpoint_name, local=True), call(name=endpoint_name, local=True)]
        )

    def test_when_not_local_with_model_not_found(self):
        # arrange
        ml_client = Mock()
        is_local_deployment = False
        endpoint_name = "test_endpoint"
        endpoint_description = "test_description"
        model_name = "test_model"
        model_version = "test_version"
        environment_image = "test_image"
        deployment_name = "test_deployment"
        compute_instance_type = "test_type"
        compute_instance_count = 2
        scoring_uri = "test_scoring_uri"
        request_in_ms = 60000

        mock_online_deployment_create_response = Mock()
        mock_online_deployment_create_response.result.return_value = None
        ml_client.online_deployments.begin_create_or_update.return_value = (
            mock_online_deployment_create_response
        )

        mock_online_endpoint_get_1 = Mock()
        mock_online_endpoint_get_1.name = endpoint_name
        mock_online_endpoint_get_1.description = endpoint_description
        mock_online_endpoint_get_2 = Mock()
        mock_online_endpoint_get_2.scoring_uri = scoring_uri
        ml_client.online_endpoints.get.side_effect = [
            mock_online_endpoint_get_1,
            mock_online_endpoint_get_2,
        ]

        # act
        actual_online_deployment, actual_online_endpoint = create_online_endpoint(
            ml_client,
            is_local_deployment,
            endpoint_name,
            endpoint_description,
            model_name,
            model_version,
            environment_image,
            deployment_name,
            compute_instance_type,
            compute_instance_count,
            request_in_ms,
        )

        # assert
        self.assertEqual(actual_online_deployment.name, deployment_name)
        self.assertEqual(actual_online_endpoint.scoring_uri, scoring_uri)

        ml_client.online_endpoints.begin_create_or_update.assert_has_calls(
            [
                call(
                    ANY,
                )
            ],
            any_order=True,
        )
        ml_client.online_deployments.begin_create_or_update.assert_called_once_with(
            deployment=ANY, local=False
        )
        ml_client.online_endpoints.get.assert_has_calls(
            [
                call(name=endpoint_name, local=False),
                call(name=endpoint_name, local=False),
            ]
        )

    def test_when_not_local_with_model_version(self):
        # arrange
        ml_client = Mock()
        is_local_deployment = False
        endpoint_name = "test_endpoint"
        endpoint_description = "test_description"
        model_name = "test_model"
        model_version = "test_version"
        environment_image = "test_image"
        deployment_name = "test_deployment"
        compute_instance_type = "test_type"
        compute_instance_count = 2
        scoring_uri = "test_scoring_uri"
        request_in_ms = 60000

        mock_online_endpoint_create_response = Mock()
        mock_online_endpoint_create_response.result.return_value = None
        ml_client.online_endpoints.begin_create_or_update.return_value = (
            mock_online_endpoint_create_response
        )

        mock_online_deployment_create_response = Mock()
        mock_online_deployment_create_response.result.return_value = None
        ml_client.online_deployments.begin_create_or_update.return_value = (
            mock_online_deployment_create_response
        )

        mock_online_endpoint = Mock()
        mock_online_endpoint.scoring_uri = scoring_uri
        ml_client.online_endpoints.get.side_effect = [
            ResourceNotFoundError("endpoint not found"),
            mock_online_endpoint,
        ]

        # act
        actual_online_deployment, actual_online_endpoint = create_online_endpoint(
            ml_client,
            is_local_deployment,
            endpoint_name,
            endpoint_description,
            model_name,
            model_version,
            environment_image,
            deployment_name,
            compute_instance_type,
            compute_instance_count,
            request_in_ms,
        )

        # assert
        self.assertEqual(actual_online_deployment.name, deployment_name)
        self.assertEqual(actual_online_endpoint.scoring_uri, scoring_uri)

        ml_client.online_endpoints.begin_create_or_update.assert_has_calls(
            [
                call(ANY, local=False),
                call(
                    ANY,
                ),
            ],
            any_order=True,
        )
        ml_client.online_deployments.begin_create_or_update.assert_called_once_with(
            deployment=ANY, local=False
        )
        ml_client.online_endpoints.get.assert_has_calls(
            [
                call(name=endpoint_name, local=False),
                call(name=endpoint_name, local=False),
            ]
        )

    def test_when_not_local_without_model_version(self):
        # arrange
        ml_client = Mock()
        is_local_deployment = False
        endpoint_name = "test_endpoint"
        endpoint_description = "test_description"
        model_name = "test_model"
        model_version = None
        environment_image = "test_image"
        deployment_name = "test_deployment"
        compute_instance_type = "test_type"
        compute_instance_count = 2
        scoring_uri = "test_scoring_uri"
        request_in_ms = 60000

        models = [Model(name=model_name, version="1.0.0", tags={"best_model": "true"})]
        ml_client.models.list.return_value = models

        mock_online_endpoint_create_response = Mock()
        mock_online_endpoint_create_response.result.return_value = None
        ml_client.online_endpoints.begin_create_or_update.return_value = (
            mock_online_endpoint_create_response
        )

        mock_online_deployment_create_response = Mock()
        mock_online_deployment_create_response.result.return_value = None
        ml_client.online_deployments.begin_create_or_update.return_value = (
            mock_online_deployment_create_response
        )

        mock_online_endpoint = Mock()
        mock_online_endpoint.scoring_uri = scoring_uri
        ml_client.online_endpoints.get.side_effect = [
            ResourceNotFoundError("endpoint not found"),
            mock_online_endpoint,
        ]

        # act
        actual_online_deployment, actual_online_endpoint = create_online_endpoint(
            ml_client,
            is_local_deployment,
            endpoint_name,
            endpoint_description,
            model_name,
            model_version,
            environment_image,
            deployment_name,
            compute_instance_type,
            compute_instance_count,
            request_in_ms,
        )

        # assert
        self.assertEqual(actual_online_deployment.name, deployment_name)
        self.assertEqual(actual_online_endpoint.scoring_uri, scoring_uri)

        ml_client.models.list.assert_called_once_with(
            name=model_name,
        )
        ml_client.online_endpoints.begin_create_or_update.assert_has_calls(
            [
                call(ANY, local=False),
                call(
                    ANY,
                ),
            ],
            any_order=True,
        )
        ml_client.online_deployments.begin_create_or_update.assert_called_once_with(
            deployment=ANY, local=False
        )
        ml_client.online_endpoints.get.assert_has_calls(
            [
                call(name=endpoint_name, local=False),
                call(name=endpoint_name, local=False),
            ]
        )

    def test_when_not_local_without_model_version_throws_exception_when_no_best_model(
        self,
    ):
        # arrange
        ml_client = Mock()
        is_local_deployment = False
        endpoint_name = "test_endpoint"
        endpoint_description = "test_description"
        model_name = "test_model"
        model_version = None
        environment_image = "test_image"
        deployment_name = "test_deployment"
        compute_instance_type = "test_type"
        compute_instance_count = 2
        request_in_ms = 60000

        models = [Model(name=model_name, version="1.0.0", tags={"best_model": "false"})]
        ml_client.models.list.return_value = models

        mock_online_endpoint_create_response = Mock()
        mock_online_endpoint_create_response.result.return_value = None
        ml_client.online_endpoints.begin_create_or_update.return_value = (
            mock_online_endpoint_create_response
        )

        mock_online_deployment_create_response = Mock()
        mock_online_deployment_create_response.result.return_value = None
        ml_client.online_deployments.begin_create_or_update.return_value = (
            mock_online_deployment_create_response
        )

        ml_client.online_endpoints.get.side_effect = [
            ResourceNotFoundError("endpoint not found")
        ]

        # act
        with self.assertRaises(Exception) as context:
            create_online_endpoint(
                ml_client,
                is_local_deployment,
                endpoint_name,
                endpoint_description,
                model_name,
                model_version,
                environment_image,
                deployment_name,
                compute_instance_type,
                compute_instance_count,
                request_in_ms,
            )

        # assert
        self.assertTrue("does not have a best model" in str(context.exception))
        ml_client.models.list.assert_called_once_with(
            name=model_name,
        )
        ml_client.online_endpoints.begin_create_or_update.assert_has_calls(
            [call(ANY, local=False)], any_order=True
        )
        ml_client.online_endpoints.get.assert_has_calls(
            [call(name=endpoint_name, local=False)]
        )

    def test_when_deploy_remote_with_inference_setting(self):
        # arrange
        ml_client = Mock()
        is_local_deployment = False
        endpoint_name = "test_endpoint"
        endpoint_description = "test_description"
        model_name = "test_model"
        model_version = "1.0.0"
        environment_image = "test_image"
        deployment_name = "test_deployment"
        compute_instance_type = "test_type"
        compute_instance_count = 2
        nms_iou_thresh = 0.0001
        tile_grid_size = [3, 2]
        scoring_uri = "test_scoring_uri"
        request_in_ms = 60000

        models = [Model(name=model_name, version="1.0.0", tags={"best_model": "true"})]
        ml_client.models.list.return_value = models

        mock_online_deployment_create_response = Mock()
        mock_online_deployment_create_response.result.return_value = None
        ml_client.online_deployments.begin_create_or_update.return_value = (
            mock_online_deployment_create_response
        )

        mock_online_endpoint_get_1 = Mock()
        mock_online_endpoint_get_1.name = endpoint_name
        mock_online_endpoint_get_1.description = endpoint_description
        mock_online_endpoint_get_2 = Mock()
        mock_online_endpoint_get_2.scoring_uri = scoring_uri
        ml_client.online_endpoints.get.side_effect = [
            mock_online_endpoint_get_1,
            mock_online_endpoint_get_2,
        ]

        # act
        deploy_mock = Mock()
        with self.assertRaises(FileNotFoundError):
            create_online_endpoint(
                ml_client,
                is_local_deployment,
                endpoint_name,
                endpoint_description,
                model_name,
                model_version,
                environment_image,
                deployment_name,
                compute_instance_type,
                compute_instance_count,
                request_in_ms,
                nms_iou_thresh=nms_iou_thresh,
                tile_grid_size=tile_grid_size,
            )

            # assert
            deploy_mock._update_inference_model.assert_has_calls(
                [call(ml_client=ANY, nms_iou_thresh=0.0001, tile_grid_size=[3, 2])]
            )
