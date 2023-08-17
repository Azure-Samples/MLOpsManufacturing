import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
from app.automl_pipeline.utils.npy_convertor import NpyConvertor


class TestNpyConvertor(unittest.TestCase):
    """
    This class will test all scenarios for NptConvertor
    """
    def test_to_jsonl(self):
        labels_map = { "1": "symbol_1", "2": "symbol_2" }

        # create npy array
        npy_array = np.array([['symbol_1', [10, 10, 20, 20], '1'], ['symbol_2', [100, 100, 200, 200], '2']], dtype=object)

        # act
        result = NpyConvertor.to_jsonl(npy_array, os.path.join(os.path.dirname(__file__), 'input_files/0.jpg'), '0.jpg', labels_map)

        # assert
        expected_result = {
            "image_url": "0.jpg",
            "image_details": {
                "format": "jpg",
                "width": 7168,
                "height": 4561
            },
            "label": [
                {
                    "label": "symbol_1",
                    "topX": 0.0013950892857142857,
                    "topY": 0.0021925016443762333,
                    "bottomX": 0.0027901785714285715,
                    "bottomY": 0.0043850032887524665
                },
                {
                    "label": "symbol_2",
                    "topX": 0.013950892857142858,
                    "topY": 0.021925016443762334,
                    "bottomX": 0.027901785714285716,
                    "bottomY": 0.04385003288752467
                }
            ]
        }
        self.assertEqual(result, expected_result)
