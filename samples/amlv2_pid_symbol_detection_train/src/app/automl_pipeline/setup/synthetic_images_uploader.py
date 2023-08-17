import os
import json
import argparse
import subprocess
from dotenv import load_dotenv
import numpy as np
from tqdm import tqdm
from automl_pipeline.utils.uploader_client import UploaderClient
from automl_pipeline.utils.zip_helper import ZipHelper
from automl_pipeline.utils.npy_convertor import NpyConvertor
from automl_pipeline.utils.file_helper import FileHelper
import distutils.dir_util

load_dotenv()
label_config_map_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'label_config_map.json')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-blob-sas-url', dest='raw_url', type=str, required=True)
    parser.add_argument('--raw-input-path', dest='raw_input_path', type=str, required=True)
    parser.add_argument('--image-output-path', dest='image_output_path', type=str, required=True)
    parser.add_argument('--label-output-path', dest='label_output_path', type=str, required=True)

    args = parser.parse_args()
    return args


def main():
    # unzip the raw input path
    args = get_args()
    raw_input_path = args.raw_input_path
    image_output_path = args.image_output_path
    label_output_path = args.label_output_path

    # this is the raw blob SAS url
    raw_url = args.raw_url

    with open(label_config_map_path, 'r') as f:
        label_config_map = json.load(f)

    storage_account_connection_string = os.getenv("STORAGE_ACCOUNT_CONNECTION_STRING")
    storage_account_container_name = os.getenv("STORAGE_ACCOUNT_CONTAINER_NAME")

    # create the output path if it doesn't exist
    FileHelper.ensure_folder_exists(image_output_path)
    FileHelper.ensure_folder_exists(label_output_path)

    # download the raw input zip file from the raw blob SAS url
    if not os.path.exists(raw_input_path):
        print(f'Downloading {raw_input_path} from {raw_url}')

        # download the raw input zip file from the raw blob SAS url
        curl_command = ['curl', '-o', raw_input_path, raw_url]

        completed_process = subprocess.run(curl_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print(completed_process.stdout.decode('utf-8'))

    # unzip the raw input zip file to a temp directory in the same folder as the raw input path
    temp_dir = os.path.join(os.path.dirname(raw_input_path), 'temp')
    print(f'Unzipping {raw_input_path} to {temp_dir}')
    ZipHelper.unzip(raw_input_path, temp_dir)
    print(f'Unzipped {raw_input_path} to {temp_dir}')

    # extract file name from raw input path
    file_extension = os.path.splitext(raw_input_path)[1]
    unarchive_folder_name = os.path.join(temp_dir, os.path.basename(raw_input_path).replace(file_extension, ''))

    # copy all images to the image output path after unzipping from the images directory in the temp directory
    images_dir = os.path.join(unarchive_folder_name, 'images')
    print(f'Copy all jpg files from {images_dir} to {image_output_path}')

    # copy all jpg files from the images directory to the image output path
    try:
        distutils.dir_util.copy_tree(images_dir, image_output_path)
        print("Copying successful")
    except Exception as e:
        print("Error:", e)

    # load all npy files containing label information and convert them to a jsonl file compatible with Azure Machine Learning Tables
    # all labels needs to be normalized to a value between 0 and 1 as per the dimensions of the image
    # the jsonl file will contain the following fields: image_url, "image_details (format, width and height) and labels (label, topX, topY,
    # bottomX, bottomY) in normalized values
    print('Converting npy files to jsonl format...')
    jsonl_file = []
    labels_dir = os.path.join(unarchive_folder_name, 'labels')

    for npy_file in tqdm(FileHelper.list_files_endwith(labels_dir, 'symbols.npy')):
        # load the npy file
        npy_array = np.load(npy_file, allow_pickle=True)

        # get the image name
        image_name = os.path.basename(npy_file).replace('_symbols.npy', '.jpg')

        # load image and get its dimenstions
        image_path = os.path.join(images_dir, image_name)

        json_line = NpyConvertor.to_jsonl(npy_array, image_path, image_name, label_config_map)

        # append the json line to the jsonl file
        jsonl_file.append(json_line)

    # write the jsonl file to the label output path
    jsonl_file_path = os.path.join(label_output_path, 'synthetic-image-annotations.jsonl')
    FileHelper.save_jsonl(jsonl_file, jsonl_file_path)

    # upload files located in the image output path to Azure Blob Storage
    uploader_client = UploaderClient(storage_account_connection_string, storage_account_container_name)
    uploader_client.upload(image_output_path, "")

    # upload files located in the label output path to Azure Blob Storage
    uploader_client.upload(label_output_path, "")


if __name__ == "__main__":
    main()
