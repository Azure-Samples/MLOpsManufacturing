"""
MltableAggregator
"""
import glob
import json
import os


class MltableAggregator:
    """Aggregates individual label MLTables into a single MLTtable."""

    def __init__(self,
                 input_image_data_path: str,
                 input_label_data_path: str,
                 input_images_string_absolute_path: str,
                 output_path: str,
                 is_fast_training: bool = False):
        self.input_image_data_path = input_image_data_path
        self.input_label_data_path = input_label_data_path
        self.input_images_string_absolute_path = \
            input_images_string_absolute_path
        self.output_path = output_path
        self.annotations_file_path = os.path.join(output_path,
                                                  "annotations.jsonl")
        self.is_fast_training = is_fast_training

    def create_aggregated_mltable_file(self):
        """
        Create the aggregated MLTable file by combining data
        from individual jsonl files.
        """

        json_lines = []
        print('Building aggregated MLTable file...')

        # Aggregate the data from all .jsonl files in the input path
        # into a single jsonl file, making sure that absolute
        # image path is set as "image_url"
        json_lines = self.aggregate_label_mltables()

        # Write the aggregated jsonl file
        with open(self.annotations_file_path, "w") as f:
            f.write("\n".join(json_lines))

        # Create and save mltable
        mltable_file_contents = self.create_ml_table_file(
            os.path.basename(self.annotations_file_path)
        )
        self.save_ml_table_file(mltable_file_contents)

    def aggregate_label_mltables(self):
        """Aggregates data from all jsonl files in input label data path."""

        json_lines = []
        current_processed_number = 0
        for file in glob.glob(os.path.join(self.input_label_data_path,
                                           "*.jsonl")):
            with open(file, encoding='utf-8') as f:
                for line in f:
                    try:
                        line = self.set_absolute_image_path_in_label(line)
                        json_lines.append(line)
                        current_processed_number += 1
                        if self.is_fast_training and \
                                current_processed_number == 20:
                            return json_lines
                    except Exception as exc:
                        print(f"Error parsing line {line}: {exc}")

        return json_lines

    def set_absolute_image_path_in_label(self, line):
        """
        Set the absolute image path in a single label
        json object as "image_url".
        """

        json_line = json.loads(line)
        image_filename = json_line["image_url"]
        image_filename = os.path.basename(image_filename)
        image_url_with_path = os.path.join(
            self.input_images_string_absolute_path,
            image_filename)  # Expected: "/path/to/X.jpg"
        json_line["image_url"] = image_url_with_path

        return json.dumps(json_line)

    def create_ml_table_file(self, filename):
        """Create ML Table definition"""

        return (
            "paths:\n"
            "  - file: ./{0}\n"
            "transformations:\n"
            "  - read_json_lines:\n"
            "        encoding: utf8\n"
            "        invalid_lines: error\n"
            "        include_path_column: false\n"
            "  - convert_column_types:\n"
            "      - columns: image_url\n"
            "        column_type: stream_info"
        ).format(filename)

    def save_ml_table_file(self, mltable_file_contents):
        """
        Save the MLTable file to the output path.
        """
        with open(os.path.join(self.output_path, "MLTable"), "w") as f:
            f.write(mltable_file_contents)
