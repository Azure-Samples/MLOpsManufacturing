"""
Registers the best model.

As output, expects the registration of the model with a new version in Azure ML and
`model_metadata.json` artifact.
"""
import argparse
import json
import os

from azureml.core import Run, Workspace

import mlflow
from mlflow.tracking.client import MlflowClient


def parse_args():
    """
    Arguments parser
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-input-path", dest="model_input_path", type=str, help="Path to input model")
    parser.add_argument("--model-base-name", dest="model_base_name", type=str, help="Name of the registered model")
    parser.add_argument("--build-source", dest="build_source", type=str, help="The build source of the pipeline")
    parser.add_argument("--build-id", dest="build_id", type=str, help="The build id of the pipeline")
    parser.add_argument("--model-metadata-path", dest="model_metadata_path", type=str, help="Path to model metadata")
    parser.add_argument("--dataset-version", dest="dataset_version", type=str, help="data set version for training")

    args = parser.parse_args()
    return args


def main(
        model_input_path: str,
        model_base_name: str,
        model_metadata_path: str,
        experiment_name: str,
        tracking_uri: str,
        build_source: str,
        build_id: str,
        dataset_version: str):
    """
    Register Model Example
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Get Run ID from model path
    print("Getting model path")
    mlmodel_path = os.path.join(model_input_path, "MLmodel")
    runid = ""
    with open(mlmodel_path, "r") as modelfile:
        for line in modelfile:
            if "run_id" in line:
                runid = line.split(":")[1].strip()

    # Construct Model URI from run ID extract previously
    model_uri = "runs:/{}/outputs/mlflow-model/".format(runid)
    print("Model URI: " + model_uri)

    # Register the model with Model URI and Name of choice
    registered_name = model_base_name
    print(f"Registering model as {registered_name}")
    model_version = mlflow.register_model(model_uri, registered_name)
    version = model_version.version

    # getting the mlflow run and adding the mean average precision to the model tags
    print("Getting run")
    run = mlflow.get_run(run_id=runid)

    model_map = run.data.metrics.get('mean_average_precision')
    model_precision = run.data.metrics.get('precision')
    model_recall = run.data.metrics.get('recall')

    print("Model map: " + str(model_map))
    print("Model precision: " + str(model_precision))
    print("Model recall: " + str(model_recall))

    client = MlflowClient()
    client.set_model_version_tag(
        name=registered_name,
        version=version,
        key="mean_average_precision",
        value=model_map,
    )

    client.set_model_version_tag(
        name=registered_name,
        version=version,
        key="recall",
        value=model_recall
    )

    client.set_model_version_tag(
        name=registered_name,
        version=version,
        key="precision",
        value=model_precision
    )

    client.set_model_version_tag(
        name=registered_name,
        version=version,
        key="best_model",
        value="false"
    )

    client.set_model_version_tag(
        name=registered_name,
        version=version,
        key="build_source",
        value=build_source
    )

    client.set_model_version_tag(
        name=registered_name,
        version=version,
        key="build_id",
        value=build_id
    )

    client.set_model_version_tag(
        name=registered_name,
        version=version,
        key="dataset_version",
        value=dataset_version
    )

    json_data = {
        'name': registered_name,
        'version': version,
        'mean_average_precision': model_map
    }

    print('Writing model metadata')
    mlmetadata_path = os.path.join(model_metadata_path, "model_metadata.json")
    with open(mlmetadata_path, 'w') as f:
        json.dump(json_data, f)


# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()
    model_input_path = args.model_input_path
    model_base_name = args.model_base_name
    model_metadata_path = args.model_metadata_path
    build_source = args.build_source
    build_id = args.build_id

    aml_context = Run.get_context()

    if hasattr(aml_context, 'experiment'):
        experiment_name = aml_context.experiment.name
        tracking_uri = aml_context.experiment.workspace.get_mlflow_tracking_uri()

        with open(args.dataset_version, "r") as f:
            dataset_version = f.read()
    else:
        experiment_name = 'local-experiment'
        dataset_version = args.dataset_version
        from automl_pipeline.config import config
        tracking_uri = Workspace(
            workspace_name=config.workspace_name,
            subscription_id=config.subscription_id,
            resource_group=config.resource_group_name
        ).get_mlflow_tracking_uri()

    # run main function
    main(
        model_input_path,
        model_base_name,
        model_metadata_path,
        experiment_name,
        tracking_uri,
        build_source,
        build_id,
        dataset_version)
