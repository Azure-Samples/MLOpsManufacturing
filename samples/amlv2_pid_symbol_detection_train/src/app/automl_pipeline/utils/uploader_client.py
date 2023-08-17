import os
import concurrent.futures
from tqdm import tqdm
from azure.storage.blob import BlobServiceClient
from automl_pipeline.utils.file_helper import FileHelper


class UploaderClient:
    '''
    A class to upload files from local machine to a blob storage
    '''

    def __init__(self, connection_string, container_name):
        service_client = BlobServiceClient.from_connection_string(connection_string, connection_pool_maxsize=10)
        self.client = service_client.get_container_client(container_name)
        self.source = ''
        self.prefix = ''

    def upload(self, source, dest):
        '''
        Upload a file or directory to a path inside the container
        '''

        if (os.path.isdir(source)):
            self._upload_dir(source, dest)
        else:
            self._upload_file(source, dest)

    def _upload_file(self, source, dest):
        '''
        Upload a single file to a path inside the container
        '''

        with open(source, 'rb') as data:
            self.client.upload_blob(name=dest, data=data, overwrite=True)

        FileHelper.delete(source)

    def _upload_file_concurrent(self, file_path: str):
        blob_path = file_path
        self._upload_file(file_path, blob_path)
        return file_path

    def _upload_dir(self, source, dest):
        '''
        Upload a directory to a path inside the container
        '''

        self.prefix = '' if dest == '' else dest + '/'
        self.prefix += os.path.basename(source) + '/'
        files = FileHelper.list_files_endwith(source, '')
        self.source = source
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        with executor:
            progress_bar = tqdm(total=len(files), desc="Uploading files")
            # Submit the tasks and update the progress bar when each task completes
            futures = [executor.submit(self._upload_file_concurrent, file_path) for file_path in files]
            for future in concurrent.futures.as_completed(futures):
                progress_bar.update(1)

        executor.shutdown(wait=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-output-path', dest='image_output_path', type=str, required=True)
    parser.add_argument('--label-output-path', dest='label_output_path', type=str, required=True)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    from automl_pipeline.config import config
    import argparse
    args = get_args()
    image_output_path = args.image_output_path
    label_output_path = args.label_output_path

    # upload files located in the image output path to Azure Blob Storage
    uploader_client = UploaderClient(config.storage_account_connection_string, config.storage_account_container_name)
    uploader_client.upload(image_output_path, "")

    # upload files located in the label output path to Azure Blob Storage
    uploader_client.upload(label_output_path, "")
