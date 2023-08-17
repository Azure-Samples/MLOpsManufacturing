"""
Tags the best model.

As output, the best performing model is tagged, so that it can be easily identified and used for
future tasks.
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

    parser.add_argument("--model-metadata-path", dest="model_metadata_path", type=str, help='Path to model metadata')
    parser.add_argument("--model-base-name", dest="model_base_name", type=str, help="Name of the registered model")
    parser.add_argument("--best-model-metadata-path", dest="best_model_metadata_path", type=str, help="The metadata of the best model")

    args = parser.parse_args()
    return args


def write_best_model_file(best_model_metadata_path: str, model_name: str, model_version: str):
    mlbest_model_metadata_path = os.path.join(best_model_metadata_path, "best_model_metadata.json")
    with open(mlbest_model_metadata_path, "w") as f:
        json.dump({
            "name": model_name,
            "version": model_version
        }, f)


def main(
    model_metadata_path: str,
    model_base_name: str,
    best_model_metadata_path: str,
    experiment_name: str,
    tracking_uri: str
):
    # Set Tracking URI
    print("tracking_uri: {0}".format(tracking_uri))
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    mlmodel_metadata_path = os.path.join(model_metadata_path, "model_metadata.json")
    with open(mlmodel_metadata_path, "r") as f:
        model_metadata = json.load(f)

    new_mean_average_precision = model_metadata["mean_average_precision"]
    new_model_version = model_metadata["version"]

    mlflow_client = MlflowClient()

    models = mlflow_client.search_model_versions(f"name = '{model_base_name}'")

    best_model = None
    for model in models:
        if 'best_model' not in model.tags:
            continue

        best_model_value = model.tags['best_model']
        if best_model_value.lower() == 'true':
            best_model = model
            break

    if best_model is None:
        print("No best model found, registering current model as best model")
        mlflow_client.set_model_version_tag(
            name=model_base_name,
            version=new_model_version,
            key="best_model",
            value="true"
        )
        write_best_model_file(best_model_metadata_path, model_base_name, new_model_version)
        return

    # check if the best model is better than the current model
    best_model_version = best_model.version
    best_model_mean_average_precision = float(best_model.tags["mean_average_precision"])

    print(f'current best model version: {best_model_version}')

    if best_model_mean_average_precision > new_mean_average_precision:
        print("Current model is not better than the best model, skipping")
        write_best_model_file(best_model_metadata_path, model_base_name, best_model_version)
        return

    print("Current model is better than the best model, updating best model")
    mlflow_client.set_model_version_tag(
        name=model_base_name,
        version=new_model_version,
        key="best_model",
        value="true"
    )
    mlflow_client.set_model_version_tag(
        name=model_base_name,
        version=best_model_version,
        key="best_model",
        value="false"
    )
    write_best_model_file(best_model_metadata_path, model_base_name, new_model_version)


if __name__ == "__main__":
    args = parse_args()

    model_metadata_path = args.model_metadata_path
    model_base_name = args.model_base_name
    best_model_metadata_path = args.best_model_metadata_path

    aml_context = Run.get_context()

    if hasattr(aml_context, 'experiment'):
        experiment_name = aml_context.experiment.name
        tracking_uri = aml_context.experiment.workspace.get_mlflow_tracking_uri()
    else:
        experiment_name = 'local-experiment'
        from automl_pipeline.config import config
        tracking_uri = Workspace(
            workspace_name=config.workspace_name,
            subscription_id=config.subscription_id,
            resource_group=config.resource_group_name
        ).get_mlflow_tracking_uri()

    main(
        model_metadata_path,
        model_base_name,
        best_model_metadata_path,
        experiment_name,
        tracking_uri)
