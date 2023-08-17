import os
import sys
import unittest
from unittest.mock import patch
from parameterized import parameterized
from argparse import ArgumentError

python_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "..",
    "app",
    "pipeline_steps",
    "data_aggregation_src",
)
sys.path.append(python_path)
from app.pipeline_steps.data_aggregation_src.main import get_args


class TestGetArgs(unittest.TestCase):
    @parameterized.expand(
        [
            ("no", False),
            ("false", False),
            ("0", False),
            ("f", False),
            ("n", False),
            ("yes", True),
            ("true", True),
            ("1", True),
            ("t", True),
            ("y", True),
        ]
    )
    def test_happy_path(self, is_fast_training_flag, expected_is_fast_training):
        # arrange
        input_image_data_path = "input_image_data_path"
        input_label_data_path = "input_label_data_path"
        input_images_string_absolute_path = "input_images_string_absolute_path"
        output_path = "output_path"

        args = [
            "prog",
            "--input-image-data-path",
            input_image_data_path,
            "--input-label-data-path",
            input_label_data_path,
            "--input-images-string-absolute-path",
            input_images_string_absolute_path,
            "--output-path",
            output_path,
            "--is-fast-training",
            is_fast_training_flag,
        ]

        with patch.object(sys, "argv", args):
            # act
            args = get_args()

        self.assertEqual(args.input_image_data_path, input_image_data_path)
        self.assertEqual(args.input_label_data_path, input_label_data_path)
        self.assertEqual(
            args.input_images_string_absolute_path, input_images_string_absolute_path
        )
        self.assertEqual(args.output_path, output_path)
        self.assertEqual(args.is_fast_training, expected_is_fast_training)

    def test_invalid_fast_training_flag_raises_exception(self):
        # arrange
        input_image_data_path = "input_image_data_path"
        input_label_data_path = "input_label_data_path"
        input_images_string_absolute_path = "input_images_string_absolute_path"
        output_path = "output_path"
        is_fast_training_flag = "invalid"

        sys.argv = [
            "--input-image-data-path",
            input_image_data_path,
            "--input-label-data-path",
            input_label_data_path,
            "--input-images-string-absolute-path",
            input_images_string_absolute_path,
            "--output-path",
            output_path,
            "--is-fast-training",
            is_fast_training_flag,
        ]

        with patch.object(sys, "argv", sys.argv):
            # act
            with self.assertRaises(SystemExit) as cm:
                _ = get_args()

        # assert
