# Samples Overview

The samples in this project are intended for providing a starting point for building ML solutions with MLOps practices.

They are meant to represent different frameworks such as Tensorflow or Yolo, different scenarios such as model training
in the cloud or influencing on the edge, or how certain technologies are used such as mocking Azure ML SDK
for unit testing or leveraging observability library to send logs and metrics to different destinations.

Each sample has a README that details what it's meant to demonstrate.
Below we provide some classifications to easily identify the samples after three aspects:

- Relation to tools from the [common](../common) section of this repo
- [MLOps / DevOps for Data Science](https://docs.microsoft.com/en-us/azure/machine-learning/concept-model-management-and-deployment)
aspects which are covered
- [Azure Well Architectured Framework](https://docs.microsoft.com/en-us/azure/architecture/framework/)
principles which are covered

## Relation to Common section

|Sample|[azureml_appinsights_logger](../common/azureml_appinsights_logger)|[infrastructure](../common/infrastructure/)|[pipeline_monitor](../common/pipeline_monitor/)/[trigger](../common/pipeline_trigger/)|[pytest-fixtures](../common/pytest-fixtures/)|
|----|----|----|----|----|
|[Appendable AML Pipeline Step Template](./appendable-template)|compatible (not implemented)|prerequisite|compatible|compatible (not implemented)|
|[Hyperparameter Tutorial](./hyperparameter-tutorial)|compatible (not implemented)|prerequisite|compatible|compatible (not implemented)|
|[Image classification with Tensorflow and Keras](./image-classification-tensorflow)|compatible (not implemented yet)|prerequisite|compatible|compatible (not implemented)|
|[Run Kaldi ASR Toolkit in AML Pipelines](./kaldi-asr-yesno)|compatible (other logger implemented)|prerequisite|compatible|compatible (unit tests differently implemented)|
|[Run non-Python code in AML Pipelines](./non-python-preprocess)|compatible (not implemented yet)|prerequisite|compatible|showcasing|
|[Parallel Data Preprocessing in AML Pipelines](./parallel-processing-california-housing-data)|compatible (not implemented)|prerequisite|compatible|compatible (not implemented)|
|[Wrapping existing ML scripts for AML Pipelines Tutorial](./wrapping-existing-ml-scripts-tutorial)|compatible (not implemented)|prerequisite|compatible|compatible (not implemented)|
|[MLOps for Azure Custom Question Answering](./mlops-for-azure-custom-question-answering)|not compatible|IaC script provided|not compatible|not compatible|
|[Edge Inferencing and MLOps](./edge-inferencing-and-mlops)|compatible (not implemented)|prerequisite|compatible|compatible|

## Relation to MLOps / DevOps for Data Science

|Sample|CI/CD (AML pipeline)|CI/CD (ML model)|Use of AML pipelines|Model Training|
|----|----|----|----|----|
|[Azure ML Python SDK v1 Basic Training](./amlv1_simple_train)|:x:|:x:|:x:|:heavy_check_mark:|
|[Azure ML Python SDK v2 P&ID Symbol Detection Training](./amlv2_pid_symbol_detection_train/)|:x:|:x:|:heavy_check_mark:|:heavy_check_mark:|
|[Appendable AML Pipeline Step Template](./appendable-template)|:x:|:x:|:heavy_check_mark:|:heavy_check_mark:|
|[Hyperparameter Tutorial](./hyperparameter-tutorial)|:x:|:x:|:heavy_check_mark:|:heavy_check_mark:|
|[Image classification with Tensorflow and Keras](./image-classification-tensorflow)|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|[Run Kaldi ASR Toolkit in AML Pipelines](./kaldi-asr-yesno)|:heavy_check_mark:|:x:|:heavy_check_mark:|:x:|
|[Run non-Python code in AML Pipelines](./non-python-preprocess)|:heavy_check_mark:|:x:|:heavy_check_mark:|:x:|
|[Parallel Data Preprocessing in AML Pipelines](./parallel-processing-california-housing-data)|:x:|:x:|:heavy_check_mark:|:heavy_check_mark:|
|[Wrapping existing ML scripts for AML Pipelines Tutorial](./wrapping-existing-ml-scripts-tutorial)|:x:|:x:|:heavy_check_mark:|:heavy_check_mark:|
|[MLOps for Azure Custom Question Answering](./mlops-for-azure-custom-question-answering)|:heavy_check_mark:|:x:|:x:|:x:|
|[Edge Inferencing and MLOps](./edge-inferencing-and-mlops)|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|

## Relation to Azure Well Architected Framework Principals

|Sample|[Cost Optimization](https://docs.microsoft.com/en-us/azure/architecture/framework/cost/overview)|[Operational Excellence](https://docs.microsoft.com/en-us/azure/architecture/framework/devops/overview)|[Performance Efficiency](https://docs.microsoft.com/en-us/azure/architecture/framework/scalability/overview)|[Reliability](https://docs.microsoft.com/en-us/azure/architecture/framework/resiliency/overview)|[Security](https://docs.microsoft.com/en-us/azure/architecture/framework/security/security-principles)|
|----|----|----|----|----|----|
|[Appendable AML Pipeline Step Template](./appendable-template)|Pay for consumption|||||
|[Edge Object Detection](./edge-object-detection)|Pay for consumption|Fully automated infrastructure provisioning and code deployment, provides guidance on design, has edge container monitoring built in|Scalable deployment of modules to many devices|Can recover from losing network for for a full day, monitoring and container health built in but would require setup to hook into alerting system||
|[Hyperparameter Tutorial](./hyperparameter-tutorial)|Pay for consumption|Optimize build and release processes||||
|[Image classification with Tensorflow and Keras](./image-classification-tensorflow)|Pay for consumption|Optimize build and release processes, Monitor the system (planned)||Monitor and measure application health (planned)|Use Identity as Primary Access Control|
|[Run Kaldi ASR Toolkit in AML Pipelines](./kaldi-asr-yesno)|Pay for consumption|Optimize build and release processes|||Use Identity as Primary Access Control|
|[Run non-Python code in AML Pipelines](./non-python-preprocess)|Pay for consumption|Optimize build and release processes, Monitor the system (planned)|||Use Identity as Primary Access Control|
|[Parallel Data Preprocessing in AML Pipelines](./parallel-processing-california-housing-data)|Pay for consumption|Monitor the system (planned)|Monitor and optimize (planned)|||
|[Wrapping existing ML scripts for AML Pipelines Tutorial](./wrapping-existing-ml-scripts-tutorial)|Pay for consumption|||||
|[MLOps for Azure Custom Question Answering](./mlops-for-azure-custom-question-answering)|Pay for consumption|Fully automated infrastructure provisioning and code deployment||||
|[Edge Inferencing and MLOps](./edge-inferencing-and-mlops)|Pay for consumption|||||
