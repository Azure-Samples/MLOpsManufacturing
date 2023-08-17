import argparse
from azureml.core import Dataset, Workspace, Run
import os
import uuid


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-input-path",
        dest="train_input_path",
        type=str,
        required=True,
        help="Path to the folder containing the images to be used for training.",
    )

    parser.add_argument(
        "--val-input-path",
        dest="val_input_path",
        type=str,
        required=True,
        help="Path to the folder containing the images to be used for validation.",
    )

    parser.add_argument(
        "--use-stratified-split",
        dest="use_stratified_split",
        type=str2bool,
        default=True,
        help="Use stratified split or random split from automl.",
    )

    parser.add_argument(
        "--output-path",
        dest="output_path",
        type=str,
        required=True,
        help="Path to the folder where the aggregated MLTable file will be saved.",
    )

    parser.add_argument(
        "--data-store-name",
        dest="data_store_name",
        type=str,
        required=True,
        help="Data store name to register data asset.",
    )

    args = parser.parse_args()
    return args


def upload_directory(data_path, datastore, target_folder):
    Dataset.File.upload_directory(
        src_dir=data_path,
        target=(datastore, target_folder),
        overwrite=True,
        show_progress=True,
    )


def mltable_from_json_lines_files(datastore, jsonl_path):
    return Dataset.Tabular.from_json_lines_files(path=(datastore, jsonl_path))


def get_uuid4():
    return str(uuid.uuid4())


def main(
    train_data_path,
    val_data_path,
    use_stratified_split,
    output_path,
    datastore,
    workspace,
):
    # Register the data asset
    print("Registering data asset...")

    # Generate a random guid to avoid collisions
    guid = get_uuid4()

    dataset_version_str = ""
    dataset_types = ["train", "val"] if use_stratified_split else ["train"]
    dataset_paths = (
        [train_data_path, val_data_path] if use_stratified_split else [train_data_path]
    )
    print(f"use_stratified_split={use_stratified_split}")
    for dataset_type, data_path in zip(dataset_types, dataset_paths):
        upload_directory(data_path, datastore, f"training-sets/{guid}/{dataset_type}")

        mltable_dataset = mltable_from_json_lines_files(
            datastore, f"training-sets/{guid}/{dataset_type}/annotations.jsonl"
        )
        dataset_name = (
            f"{dataset_type}-data-set-splitted"
            if use_stratified_split
            else f"{dataset_type}-data-set"
        )
        dataset = mltable_dataset.register(
            workspace,
            name=dataset_name,
            description=f"{dataset_type} data set splitted",
            create_new_version=True,
        )
        dataset_version_str = (
            dataset_version_str
            + f"{dataset_type}: azureml:{dataset.name}:{dataset.version}\t"
        )

    with open(output_path, "w") as f:
        f.write(dataset_version_str)


if __name__ == "__main__":
    args = get_args()

    train_data_path = os.path.join(args.train_input_path)
    val_data_path = os.path.join(args.val_input_path)
    output_path = args.output_path
    data_store_name = args.data_store_name
    use_stratified_split = args.use_stratified_split

    aml_context = Run.get_context()

    if hasattr(aml_context, "experiment"):
        workspace = aml_context.experiment.workspace
    else:
        from automl_pipeline.config import config

        workspace = Workspace(
            workspace_name=config.workspace_name,
            subscription_id=config.subscription_id,
            resource_group=config.resource_group_name,
        )

    datastore = workspace.datastores[data_store_name]

    main(
        train_data_path,
        val_data_path,
        use_stratified_split,
        output_path,
        datastore,
        workspace,
    )
