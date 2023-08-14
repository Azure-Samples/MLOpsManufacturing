import os
import logging
import subprocess


class ZipHelper:
    """A class to work with zip/tar files"""

    @staticmethod
    def unzip(input_path: str, output_path: str):
        """
        unzip the specified zip/tar file to the output path. The file can be zip or gz.
        """
        # unzip the raw input zip file to a temp directory in the same folder as the raw input path
        logging.info(f'Unzipping {input_path} to {output_path}')
        temp_dir = output_path
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        if input_path.endswith('.zip'):
            subprocess.run(['unzip', '-o', '-q', input_path, '-d', temp_dir])
        elif input_path.endswith('.tar.gz'):
            subprocess.run(['tar', '-xzf', input_path, '-C', temp_dir])
        else:
            logging.error(f'Unsupported file type: {input_path}')
            return
        logging.info(f'Unziped {input_path} to {output_path}')
