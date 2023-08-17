import os
import tempfile
import argparse
import json
from typing import List, Optional, Union

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    OnlineRequestSettings,
    Model,
    Environment,
    CodeConfiguration,
)
from azure.core.exceptions import ResourceNotFoundError
from azure.ai.ml.exceptions import LocalEndpointNotFoundError
from azure.identity import DefaultAzureCredential
import datetime


file_dir_path = os.path.dirname(__file__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subscription-id",
        dest="subscription_id",
        type=str,
        required=True,
        help="The id of the subscription",
    )
    parser.add_argument(
        "--resource-group-name",
        dest="resource_group_name",
        type=str,
        required=True,
        help="The name of the resource group",
    )
    parser.add_argument(
        "--workspace-name",
        dest="workspace_name",
        type=str,
        required=True,
        help="The name of the AML workspace",
    )
    parser.add_argument(
        "--is-local-deployment",
        dest="is_local_deployment",
        action="store_true",
        help="Flag to indicate if the deployment is local or not",
    )
    parser.add_argument(
        "--endpoint-name",
        dest="endpoint_name",
        type=str,
        default="pid-symbol-detection-endpoint",
        help="The name of the endpoint",
    )
    parser.add_argument(
        "--endpoint-description",
        dest="endpoint_description",
        type=str,
        default="The PID symbol detection endpoint",
        help="The description of the endpoint",
    )
    parser.add_argument(
        "--model-name",
        dest="model_name",
        type=str,
        help="The name of the model (required if --is-local-deployment is false)",
    )
    parser.add_argument(
        "--model-version",
        dest="model_version",
        type=str,
        help="The version of the model",
    )
    parser.add_argument(
        "--environment-image",
        dest="environment_image",
        type=str,
        required=True,
        help="The environment image",
    )
    parser.add_argument(
        "--compute-instance-type",
        dest="compute_instance_type",
        type=str,
        default="Standard_DS3_v2",
        help="The compute instance type",
    )
    parser.add_argument(
        "--compute-instance-count",
        dest="compute_instance_count",
        type=int,
        default=1,
        help="The compute instance count",
    )
    parser.add_argument(
        "--request-timeout-ms",
        dest="request_timeout_ms",
        type=int,
        default=60000,
        help="The request timeout in ms",
    )
    parser.add_argument(
        "--box-score-thresh",
        dest="box_score_thresh",
        required=False,
        type=float,
        help="during inference, only return proposals with a score greater than box_score_thresh",
    )
    parser.add_argument(
        "--nms-iou-thresh",
        dest="nms_iou_thresh",
        type=float,
        required=False,
        help="IOU threshold used during inference in nms",
    )
    parser.add_argument(
        "--box-detections-per-img",
        dest="box_detections_per_img",
        type=int,
        required=False,
        help="Maximum number of detections per image, for all classes",
    )
    parser.add_argument(
        "--tile-grid-size",
        dest="tile_grid_size",
        required=False,
        default="null",
        type=str,
        help="The grid size to use for tiling each image, an example of the value is '3x2'",
    )
    parser.add_argument(
        "--tile-overlap-ratio",
        dest="tile_overlap_ratio",
        required=False,
        type=float,
        help="Overlap ratio between adjacent tiles in each dimension",
    )
    parser.add_argument(
        "--tile-predictions-nms-threshold",
        dest="tile_predictions_nms_threshold",
        required=False,
        type=float,
        help="The IOU threshold to use to perform NMS while merging tiles",
    )

    return parser.parse_args()


def _get_best_model(ml_client: MLClient, model_name: str):
    models = ml_client.models.list(name=model_name)

    # get the model that has the best_model tag is true
    best_model = None
    for model in models:
        if model.tags.get("best_model") == "true":
            best_model = model
            break
    return best_model


def _update_inference_model(
    ml_client: MLClient, model: Model, model_name: str, model_version: str, **kwargs
) -> Model:
    """register a new model with changed inference time setting

    Args:
        ml_client (MLClient): ML Client targeted
        model (Model): original model to ne updated
        model_name (str): name of the model targeted
        model_version (str): version of of the model targeted

    Returns:
        Model: updated model registered
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        ml_client.models.download(
            name=model_name,
            version=model_version,
            download_path=tmpdir,
        )

        model_path = os.path.join(tmpdir, model_name, "mlflow-model")
        setting_path = os.path.join(model_path, "artifacts/settings.json")

        model_settings = None
        with open(setting_path) as json_file:
            model_settings = json.load(json_file)

        for arg_name in kwargs:
            if kwargs[arg_name] is not None:
                model_settings[arg_name] = kwargs[arg_name]
        with open(setting_path, "w") as fp:
            json.dump(model_settings, fp)

        model_tags = model.tags
        model_tags["base_model"] = f"{model_name}:{model_version}"

        update_model = Model(
            path=model_path,
            name=model_name,
            type=AssetTypes.MLFLOW_MODEL,
            tags=model_tags,
            properties=model.properties,
        )
        return ml_client.models.create_or_update(model=update_model)


def create_online_endpoint(
    ml_client: MLClient,
    is_local_deployment: bool,
    endpoint_name: str,
    endpoint_description: str,
    model_name: str,
    model_version: str,
    environment_image: str,
    deployment_name: str,
    compute_instance_type: str,
    compute_instance_count: int,
    request_timeout_ms: int,
    box_score_thresh: Optional[float] = None,
    nms_iou_thresh: Optional[float] = None,
    box_detections_per_img: Optional[int] = None,
    tile_grid_size: Optional[Union[List[int], str]] = None,
    tile_overlap_ratio: Optional[float] = None,
    tile_predictions_nms_thresh: Optional[float] = None,
):
    print("Creating or getting the online endpoint...")
    online_endpoint = None
    try:
        online_endpoint = ml_client.online_endpoints.get(
            name=endpoint_name, local=is_local_deployment
        )
    except (ResourceNotFoundError, LocalEndpointNotFoundError):
        online_endpoint = ManagedOnlineEndpoint(
            name=endpoint_name, description=endpoint_description, auth_mode="key"
        )

        online_endpoint_result = ml_client.online_endpoints.begin_create_or_update(
            online_endpoint, local=is_local_deployment
        )

        if not is_local_deployment:
            online_endpoint_result.result()

    model = None
    if is_local_deployment is True:
        model = Model(
            path=os.path.join(file_dir_path, "mlflow-model"),
            type=AssetTypes.MLFLOW_MODEL,
            name="local-model",
            description="Local deployment object detection model.",
        )
    elif model_version:
        model = ml_client.models.get(name=model_name, version=model_version)
    else:
        model = _get_best_model(ml_client, model_name)
        if model is None:
            raise Exception(f"Model {model_name} does not have a best model.")

    if is_local_deployment is False:
        inference_settings = [
            box_score_thresh,
            nms_iou_thresh,
            box_detections_per_img,
            tile_grid_size,
            tile_overlap_ratio,
            tile_predictions_nms_thresh,
        ]
        if any(setting is not None for setting in inference_settings):
            model = _update_inference_model(
                ml_client=ml_client,
                model=model,
                model_name=model_name,
                model_version=model_version,
                box_score_thresh=box_score_thresh,
                nms_iou_thresh=nms_iou_thresh,
                box_detections_per_img=box_detections_per_img,
                tile_grid_size=tile_grid_size,
                tile_overlap_ratio=tile_overlap_ratio,
                tile_predictions_nms_thresh=tile_predictions_nms_thresh,
            )

    environment = Environment(
        image=environment_image,
    )

    print("Online endpoint created successfully.")
    print(f"Online endpoint name: {online_endpoint.name}")
    print(f"Online endpoint description: {online_endpoint.description}")

    print("Creating the online deployment...")
    online_deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=model,
        environment=environment,
        code_configuration=CodeConfiguration(
            code=os.path.join(file_dir_path, "scoring"),
            scoring_script="online_endpoint_score.py",
        ),
        instance_type=compute_instance_type,
        instance_count=compute_instance_count,
        egress_public_network_access="disabled",
        request_settings=OnlineRequestSettings(
            request_timeout_ms=request_timeout_ms,
        ),
    )

    online_deployment_result = ml_client.online_deployments.begin_create_or_update(
        deployment=online_deployment, local=is_local_deployment
    )

    if not is_local_deployment:
        online_deployment_result.result()
        online_endpoint.traffic = {deployment_name: 100}
        ml_client.online_endpoints.begin_create_or_update(online_endpoint).result()

    online_endpoint = ml_client.online_endpoints.get(
        name=endpoint_name, local=is_local_deployment
    )
    return online_deployment, online_endpoint


if __name__ == "__main__":
    # getting the args
    args = get_args()
    subscription_id = args.subscription_id
    resource_group_name = args.resource_group_name
    workspace_name = args.workspace_name
    is_local_deployment = args.is_local_deployment
    endpoint_name = args.endpoint_name
    endpoint_description = args.endpoint_description
    model_name = args.model_name
    model_version = args.model_version
    environment_image = args.environment_image
    compute_instance_type = args.compute_instance_type
    compute_instance_count = args.compute_instance_count
    request_timeout_ms = args.request_timeout_ms
    deployment_name = f'deploy-{datetime.datetime.now().strftime("%m%d%H%M%f")}'
    box_score_thresh = args.box_score_thresh
    nms_iou_thresh = args.nms_iou_thresh
    box_detections_per_img = args.box_detections_per_img
    tile_grid_size_str = args.tile_grid_size
    if isinstance(tile_grid_size_str, str) and tile_grid_size_str == "null":
        tile_grid_size = None
    else:
        tile_grid_size_strs = str(tile_grid_size_str).replace(" ", "").split("x")
        if len(tile_grid_size_strs) == 2:
            tile_grid_size = [int(tile_grid_size_strs[0]), int(tile_grid_size_strs[1])]
        else:
            raise ValueError(
                "error while parsing tile_grid_size of {args.tile_grid_size}"
            )
    tile_overlap_ratio = args.tile_overlap_ratio
    tile_predictions_nms_thresh = args.tile_overlap_ratio

    if is_local_deployment is False and not model_name:
        raise Exception("The model name is required for non-local deployments")

    if is_local_deployment is True:
        print("creating a local deployment...")
    else:
        print("creating a deployment in AML...")

    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential, subscription_id, resource_group_name, workspace_name
    )

    online_deployment, online_endpoint = create_online_endpoint(
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
        request_timeout_ms,
        box_score_thresh,
        nms_iou_thresh,
        box_detections_per_img,
        tile_grid_size,
        tile_overlap_ratio,
        tile_predictions_nms_thresh,
    )

    print("Online deployment created successfully.")
    print(f"Online deployment name: {online_deployment.name}")
    print(f"Online endpoint scoring uri: {online_endpoint.scoring_uri}")
