import os
from dotenv import load_dotenv
import mlflow
from azureml.core.authentication import AzureCliAuthentication
from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig, Dataset, Datastore

# read environment variables from .env
load_dotenv()
subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
resource_group = os.getenv("AZURE_RESOURCE_GROUP")
workspace = os.getenv("AZURE_ML_WORKSPACE")
training_cluster = os.getenv("AZURE_ML_TRAINING_CLUSTER")
datastore_name = os.getenv("AZURE_ML_DATASTORE")
experiment_name = os.getenv("EXPERIMENT_NAME")
model_name = os.getenv("MODEL_NAME")

# connect to workspace using azure-cli credentials
cli_auth = AzureCliAuthentication()
ws = Workspace.get(name=workspace, subscription_id=subscription_id, resource_group=resource_group, auth=cli_auth)

# configure input dataset, training cluster, and experiment
training_env = Environment.get(workspace=ws, name="AzureML-lightgbm-3.2-ubuntu18.04-py37-cpu")
experiment = Experiment(workspace=ws, name=experiment_name)
datastore = Datastore.get(workspace=ws, datastore_name= datastore_name)
dataset = Dataset.File.from_files(path=[(datastore, 'iris')])

mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
mlflow.set_experiment(experiment.name)

# submit training job
config = ScriptRunConfig(
    source_directory='./train',
    script='train.py',
    arguments=[
        '--input-data', dataset.as_mount(),
    ],
    compute_target=training_cluster,
    environment=training_env)
run = experiment.submit(config)
run.wait_for_completion(show_output=True)

# register the model from the experiment
model_uri = f"runs:/{run.id}/model"
model = mlflow.register_model(model_uri, model_name)
