import os
import json


class FileHelper:
    """
    This class mainly work with OS file system
    """

    @staticmethod
    def list_files_endwith(directory: str, endswith: str):
        """
        listing all files that their names end with the specified postfix
        """
        file_list = []
        for root, _, files in os.walk(directory):
            for name in files:
                if not name.endswith(endswith):
                    continue

                file = os.path.join(root, name)

                file_list.append(file)

        return file_list

    @staticmethod
    def save_jsonl(jsonl_array, jsonl_file_path):
        """
        Save an array to the specified file in the jsonl format
        """
        with open(jsonl_file_path, 'w') as f:
            for json_line in jsonl_array:
                f.write(json.dumps(json_line) + '\r\n')

    @staticmethod
    def ensure_folder_exists(directory: str):
        """
        Making sure if the specified directory exists. If not, it will create the directory
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def delete(file_path: str):
        """
        Deleting a file
        """

        # Check if the file exists
        if os.path.exists(file_path):
            # Delete the file
            os.remove(file_path)
