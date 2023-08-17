# AML Model Online Endpoint Deployment

This document outlines the process of creating an online endpoint in an Azure Machine Learning (AML) instance that is within a private network.

## Getting Started

This constraint disables the use of any of the images from the Microsoft Container Registry and forces custom images to be used in an Azure Container Registry (ACR) instance that is linked to the AML workspace.
Microsoft provides some documentation on this issue which can be found [here](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-secure-online-endpoint?tabs=cli%2Cmodel#outbound-resource-access).
The solution in place to solve this issue is a multi-step process:

1. Create a Docker image that can manage inferencing in AML
1. Create a file that will perform inferencing (this is called the `score.py` file in AML)
1. Create an easy-to-use program to deploy the endpoint to AML

All the code for the deployment of the online endpoint is under the `src/app/automl_pipeline/deploy`.

### The Docker Image

Creating the docker image is the first step to deploying the custom scoring/inferencing to AML.
This document won't go too in-depth into the Dockerfiles that have been created but will focus on how they should be used.
Once the decision has been made to leverage either a GPU or CPU compute instance for the endpoint, the corresponding Dockerfile will need to be built and pushed into the ACR instance that is connected with the AML workspace.
This can be done with the following:

```bash
# Assuming in the src/app/automl_pipeline/deploy/env_config/cpu directory for dev.

docker build -t inference-container-cpu:<version> .
docker tag inference-container-cpu:<version> <acr>/inference-container-cpu:<version>
docker push <acr>/inference-container-cpu:<version>
```

It is recommended to use `cpu` directory for dev environments. For prod & qa environments, use `gpu` directory and encourage to replace
the container name to `inference-container-gpu` here and the next commands through this document.

Notes:

- You may need to authenticate to Azure before pushing the container to the Azure container registry:

    ```bash
    az acr login -n <acr>
    ````

- Use the FQDN as `<acr>` in the CLI commands. i.e. use `crmlwapmpnideqagrp01.azurecr.io` instead of `crmlwapmpnideqagrp01`.

### The Scoring

To perform inferencing with a custom Docker image, a scoring file is needed.
Information on creating a `score.py` file can be found [here](https://learn.microsoft.com/en-us/azure/machine-learning/v1/how-to-deploy-advanced-entry-script?view=azureml-api-1).
The scoring file for this project closely follows the scoring script from the `Binary data` section of the document referenced in the previous sentence.
Every `score.py` file needs to do a few things:

1. Load the trained model
2. Perform inferencing on the input data with the trained model

The scoring file used to perform inferencing in this project can be found [here](../src/app/automl_pipeline/deploy/scoring/online_endpoint_score.py).

### The Deployment

There is a program to deploy the scoring file to the online endpoint.
To run the file, the image must exist in the ACR instance that is linked with the AML workspace.
For more information on pushing the image to ACR please go back to the [docker image section](#the-docker-image).
Once the image is in the ACR instance, run the [deployment script](../src/app/automl_pipeline/deploy/main.py).

#### Local Deployment

Below is an example of how to run the inferencing locally.
A copy of the mlflow model will also need to be saved under the `src/app/automl_pipeline/deploy/` directory.

```bash
# assuming in the src/app directory

python -m automl_pipeline.deploy.main \
    --subscription-id <subcription-id> \
    --resource-group-name <name-of-resource-group> \
    --workspace-name <name-of-aml-workspace> \
    --is-local-deployment \
    --environment-image <acr>/inference-container-cpu:<version>
```

#### AML Deployment

Below is an example of how to deploy the model to AML.

```bash
# assuming in the src/app directory

python -m automl_pipeline.deploy.main \
    --subscription-id <subcription-id> \
    --resource-group-name <name-of-resource-group> \
    --workspace-name <name-of-aml-workspace> \
    --environment-image <acr>/inference-container-cpu:<version> \
    --model-name <name-of-model-in-aml-to-deploy> \
    --model-version <version-of-model-in-aml-to-deploy> \
    --endpoint-name <aml-endpoint-name-use-your-own>
```

AML Deployment allows to configure the model inference-time parameters, which can be different from the default values fixed during training. The configurable inference-time parameters includes:

- box_score_thresh: during inference, only return proposals with a score greater than box_score_thresh
- nms_iou_thresh: IOU threshold used during inference in nms
- box_detections_per_img: maximum number of detections per image, for all classes
- tile_grid_size: the grid size to use for tiling each image
- tile_overlap_ratio: overlap ratio between adjacent tiles in each dimension
- tile_predictions_nms_thresh: the IOU threshold to use to perform NMS while merging tiles"

Below is an example of how to deploy the model to AML, which uses customized values of box_detections_per_img and tile_grid_size.

```bash
# assuming in the src/app directory

python -m automl_pipeline.deploy.main \
    --subscription-id <subcription-id> \
    --resource-group-name <name-of-resource-group> \
    --workspace-name <name-of-aml-workspace> \
    --environment-image <acr>/inference-container-cpu:<version> \
    --model-name <name-of-model-in-aml-to-deploy> \
    --model-version <version-of-model-in-aml-to-deploy> \
    --endpoint-name <aml-endpoint-name-use-your-own>
    --box-detections-per-img 150 \
    --tile-grid-size 2x2 \
```

#### AML Deployment using Docker AML environment

Below is an example of how to deploy the model to AML using deployment AML environment.

```bash
# create deployment AML environment
docker build -t apmpnidesymboldetection-test:latest src/

# run deployment command inside of the container
docker run -it apmpnidesymboldetection-test:latest /bin/bash -c "az login && python -m automl_pipeline.deploy.main \
    --subscription-id <subcription-id> \
    --resource-group-name <name-of-resource-group> \
    --workspace-name<name-of-aml-workspace> \
    --environment-image <acr>/inference-container-cpu:<version> \ \
    --model-name <name-of-model-in-aml-to-deploy> \
    --model-version <version-of-model-in-aml-to-deploy> \
    --endpoint-name <aml-endpoint-name-use-your-own>
```

#### Running the Endpoint

There are two options to run the endpoint:

1. Use [the postman collection](./assets/InferenceEndpoint.postman_collection.json). This option returns the json representation of the bounding boxes.
    When using the postman collection, update the variables to use an actual `API Key`.
    Update the request body to send an image from your local computer.

1. Use [the show inference bounding boxes sample](../src/samples/show_inference_bounding_boxes.py). Documentation for this script can be found [here](../src/samples/README.md).
