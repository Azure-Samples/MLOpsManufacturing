import json
import unittest
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
from app.pipeline_steps.data_aggregation_src.utils.mltable_aggregator import MltableAggregator

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class TestMltableAggregator(unittest.TestCase):

    def test_set_absolute_image_path_in_label(self):

        # Arrange
        input_image_data_path = "/path/to/images"
        input_label_data_path = "/path/to/labels"
        input_images_string_absolute_path = "absolute/path/to/images"
        output_path = "/path/to/output"

        mltable_aggregator = MltableAggregator(input_image_data_path, input_label_data_path, input_images_string_absolute_path, output_path)

        line = "{\"image_url\": \"X.jpg\"}"

        # Act
        actual_line = mltable_aggregator.set_absolute_image_path_in_label(line)

        # Assert
        file_path = os.path.join("absolute/path/to/images", "X.jpg")
        expected_line = '{"image_url": "' + file_path + '"}'
        self.assertEqual(actual_line, expected_line)

    def test_aggregate_label_mltables(self):

        # Arrange
        input_image_data_path = os.path.join(THIS_DIR, 'test_data/images/')
        input_label_data_path = os.path.join(THIS_DIR, 'test_data/labels/')
        input_images_string_absolute_path = os.path.join(THIS_DIR, 'test_data/images/')
        output_path = "test_data/output"

        mltable_aggregator = MltableAggregator(input_image_data_path, input_label_data_path, input_images_string_absolute_path, output_path)

        # Act
        actual_json_lines = mltable_aggregator.aggregate_label_mltables()

        # Assert
        expected_json_line_0 = {
            "image_url": os.path.join(input_images_string_absolute_path, '0.jpg'),
            "image_details": {
                "format": "jpg",
                "width": 7168,
                "height": 4561
            },
            "label": [
                {
                    "label": "1",
                    "topX": 0.0013950892857142857,
                    "topY": 0.0021925016443762333,
                    "bottomX": 0.0027901785714285715,
                    "bottomY": 0.0043850032887524665
                },
                {
                    "label": "2",
                    "topX": 0.013950892857142858,
                    "topY": 0.021925016443762334,
                    "bottomX": 0.027901785714285716,
                    "bottomY": 0.04385003288752467
                }
            ]
        }

        expected_json_line_1 = {
            "image_url": os.path.join(input_images_string_absolute_path, '1.jpg'),
            "image_details": {
                "format": "jpg",
                "width": 7168,
                "height": 4561
            },
            "label": [
                {
                    "label": "1",
                    "topX": 0.0013950892857142857,
                    "topY": 0.0021925016443762333,
                    "bottomX": 0.0027901785714285715,
                    "bottomY": 0.0043850032887524665
                },
                {
                    "label": "2",
                    "topX": 0.013950892857142858,
                    "topY": 0.021925016443762334,
                    "bottomX": 0.027901785714285716,
                    "bottomY": 0.04385003288752467
                }
            ]
        }

        expected_json_lines = [json.dumps(expected_json_line_0), json.dumps(expected_json_line_1)]

        self.assertEqual(len(actual_json_lines), len(expected_json_lines))
        self.assertTrue(line in actual_json_lines for line in expected_json_lines)


if __name__ == '__main__':
    unittest.main()
