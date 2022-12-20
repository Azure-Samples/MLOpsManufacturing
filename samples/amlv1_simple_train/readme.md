# Overview

This example demonstrates how to use Azure ML Python SDK v1 to train a basic model on an AML dataset in a compute cluster.

__What does this sample demonstrate__:

* The concept of the [script that orchestrate a training job](main.py) in Azure ML vs. the [training script](./train/train.py) that isn't aware it's running in Azure ML.
* [Mount a dataset](main.py#L35) that references files in Azure Blob Storage to the compute cluster for training.
* Use a [curated Azure ML training environment](main.py#L22), which is different from the [environment required to run the orchestration script](requirements.txt).
* Track training experimentation using MLFlow.
* [Register the model in MLFlow](main.py#L44) model format.

__What doesn't this sample demonstrate__:

* Azure ML pipeline
* Comprehensive unit and integration tests
* Model inferencing
