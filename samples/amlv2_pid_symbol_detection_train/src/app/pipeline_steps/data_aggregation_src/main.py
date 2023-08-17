"""
data_aggregation_src/main.py
"""
import argparse
from utils.mltable_aggregator import MltableAggregator


def str2bool(v):
    """
    method to convert string to boolean
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args():
    """
    read the arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-image-data-path",
        dest="input_image_data_path",
        type=str,
        required=True,
        help="Path to the folder containing the images to be used for \
        training and testing.",
    )
    parser.add_argument(
        "--input-label-data-path",
        dest="input_label_data_path",
        type=str,
        required=True,
        help="Path to the folder containing the MLTable label \
            files to be used for training and testing.",
    )
    parser.add_argument(
        "--input-images-string-absolute-path",
        dest="input_images_string_absolute_path",
        type=str,
        required=True,
        help="Absolute AML path to the folder containing the MLTable \
            label files to be used for training and testing.",
    )
    parser.add_argument(
        "--output-path",
        dest="output_path",
        type=str,
        required=True,
        help="Path to the folder where the aggregated \
            MLTable file will be saved.",
    )
    parser.add_argument(
        "--is-fast-training",
        dest="is_fast_training",
        type=str2bool,
        required=True,
        help="Fast training flag.",
    )

    args = parser.parse_args()
    return args


def main():
    """
    main method
    """
    args = get_args()

    input_image_data_path = args.input_image_data_path
    input_label_data_path = args.input_label_data_path
    input_images_string_absolute_path = args.input_images_string_absolute_path
    output_path = args.output_path
    is_fast_training = args.is_fast_training

    # Create the annotations file
    print("Building annotations file...")
    mltable_aggregator = MltableAggregator(
        input_image_data_path,
        input_label_data_path,
        input_images_string_absolute_path,
        output_path,
        is_fast_training,
    )
    mltable_aggregator.create_aggregated_mltable_file()


if __name__ == "__main__":
    main()
