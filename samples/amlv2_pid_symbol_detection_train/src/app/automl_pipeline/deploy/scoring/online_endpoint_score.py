# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import os
import mlflow.pyfunc
import base64
import pandas as pd
import json
import logging
from azureml.contrib.services.aml_request import rawhttp
from azureml.contrib.services.aml_response import AMLResponse
from azureml.contrib.services.aml_request import AMLRequest


TASK_TYPE = 'image-object-detection'
IMAGE_FILE_KEY = 'image'


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
model = None


class RequestMethods:
    GET = 'GET'
    POST = 'POST'


def init():
    global model

    # Set up logging
    azure_model_dir = os.path.join(os.getenv('AZUREML_MODEL_DIR'))
    mlflow_dir = os.path.join(azure_model_dir, 'mlflow-model')

    model_path = None
    if not os.path.exists(mlflow_dir):
        logger.info('Model folder does not exist at: {}.'.format(mlflow_dir))
        model_path = azure_model_dir
    else:
        logger.info('Model folder exists at: {}.'.format(mlflow_dir))
        model_path = mlflow_dir

    try:
        logger.info("Loading model from path: {}.".format(model_path))
        model = mlflow.pyfunc.load_model(model_path)
        logger.info("Loading successful.")
    except Exception as e:
        logger.error("Loading failed: {}.".format(e))
        raise


def _create_error_message(message: str):
    return {
        "message": message
    }


@rawhttp
def run(request: AMLRequest):
    logger.info("Request: [{0}]".format(request))
    if request.method == RequestMethods.GET:
        response_body = str.encode(request.full_path)
        return AMLResponse(response_body, 200)

    elif request.method == RequestMethods.POST:
        if IMAGE_FILE_KEY not in request.files:
            return AMLResponse(
                _create_error_message('No image found in the request. Please send the request with an image that has a key of "image".'),
                400,
                json_str=True
            )

        logger.info("The request contains a valid file... Passing the request to the model...")
        file_bytes = request.files[IMAGE_FILE_KEY]
        request_df = pd.DataFrame(
            data=[base64.encodebytes(file_bytes.read()).decode('utf-8')],
            columns=["image"],
        )

        result = model.predict(request_df).to_json(orient='records')
        result = json.loads(result)[0]

        logger.info("Finished running inference on the image.")
        return AMLResponse(result, 200, json_str=True)
