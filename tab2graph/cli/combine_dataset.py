import typer
import logging
import os
import copy
import numpy as np
import shutil

import dbinfer_bench as dbb

from .. import yaml_utils
from typing import List
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


def combine_dataset(
    dataset_paths_str: str = typer.Argument(
        ...,
        help="List of paths to the dataset folders, divided by ','",
    ),
    combined_dataset_name: str = typer.Argument(..., help="Combined dataset name"),
    output_path_str: str = typer.Argument(
        ..., help="Output path for the generated dataset."
    ),
    mode: str = typer.Option(
        "--mode", help="Mode of the dataset combination. 'file' or 'folder'."
    ),
):
    # Read several yamls from these datasets
    logger.info("Reading and combining yamls.")
    dataset_paths = dataset_paths_str.split(",")
    if mode == "folder":
        # each dataset_path is a folder, containing more than one dataset
        datasets_paths = []
        for dataset_path in dataset_paths:
            for item in os.listdir(dataset_path):
                datasets_paths.append(os.path.join(dataset_path, item))
        dataset_paths = datasets_paths
    output_path_origin = Path(output_path_str).resolve()
    if not output_path_origin.exists():
        os.makedirs(output_path_origin)
    else:
        # delete the existing files in the output path
        shutil.rmtree(output_path_origin)
    # output_path_after=os.path.join(output_path_origin,combined_dataset_name)
    # os.makedirs(output_path_after, exist_ok=True)
    # output_path = Path(output_path_after).resolve()
    os.makedirs(output_path_origin, exist_ok=True)
    output_path = Path(output_path_origin).resolve()

    metadatas = []
    for dataset_path in dataset_paths:
        dataset_Path = Path(dataset_path).resolve()
        metadata_path = dataset_Path / "metadata.yaml"
        if metadata_path.exists():
            config = yaml_utils.load_pyd(dbb.DBBRDBDatasetMeta, metadata_path)
            # lowercase the dataset name
            config.dataset_name = config.dataset_name.lower()
            metadatas.append(config)

    # then combine them to one yaml
    combined_dataset_meta = combine_metadata_files(metadatas, combined_dataset_name)
    output_yaml_path_str = os.path.join(output_path, "metadata.yaml")

    output_yaml_path = Path(output_yaml_path_str).resolve()
    yaml_utils.save_pyd(combined_dataset_meta, output_yaml_path)
    # Copy the data files and task files under one directory. (Can reference to generate_joint_v4-nomic-clustering.sh)
    logger.info("Loading data.")
    combine_and_move_files(dataset_paths, metadatas, output_path)

    logger.debug(config.json())

    logger.info(f"Creating new combined dataset {combined_dataset_name}.")


# Given some metadatas of datasets, output a metadata for the combined dataset.
def combine_metadata_files(
    metadatas: List, combined_dataset_name: str
) -> dbb.DBBRDBDatasetMeta:
    combined_tables = []
    combined_tasks = []
    for metadata in metadatas:
        for table in metadata.tables:
            new_table = copy.deepcopy(table)
            new_table.name = f"{metadata.dataset_name}-{table.name}"
            source_name_part = table.source.split("/")[1]
            new_table.source = f"data/{metadata.dataset_name}-{source_name_part}"
            # Check all foreign keys and update them
            for column in new_table.columns:
                if column.dtype == "foreign_key":
                    column.link_to = f"{metadata.dataset_name}-{column.link_to}"
            combined_tables.append(new_table)
        for task in metadata.tasks:
            new_task = copy.deepcopy(task)
            new_task.name = f"{metadata.dataset_name}-{task.name}"
            new_task.source = f"{metadata.dataset_name}-{task.source}"
            new_task.target_table = f"{metadata.dataset_name}-{task.target_table}"
            combined_tasks.append(new_task)
    combined_dataset_meta = dbb.DBBRDBDatasetMeta(
        dataset_name=combined_dataset_name, tables=combined_tables, tasks=combined_tasks
    )
    return combined_dataset_meta


def combine_and_move_files(
    dataset_paths: List[str], metadatas: List, output_path: Path
):
    # transfer str to Path object
    if not output_path.exists():
        os.makedirs(output_path)

    # create output_path/data
    data_folder = os.path.join(output_path, "data")
    os.makedirs(data_folder, exist_ok=True)

    for dataset_path_str, metadata in zip(dataset_paths, metadatas):
        dataset_path = Path(dataset_path_str).resolve()

        # fetch the name of the current dataset
        # dataset_name = dataset_path.name
        dataset_name = metadata.dataset_name
        # deal with data
        source_data_folder = dataset_path / "data"
        if source_data_folder.exists() and source_data_folder.is_dir():
            for item in source_data_folder.iterdir():
                # copy the items to output_path/data
                src_item = os.path.join(source_data_folder, item.name)
                dst_item_name = f"{dataset_name}-{item.name}"
                dst_item = os.path.join(data_folder, dst_item_name)
                if os.path.isdir(src_item):
                    shutil.copytree(src_item, dst_item)
                else:
                    shutil.copy2(src_item, dst_item)
        else:
            print(f"Data folder not found in {dataset_path}. Skipping.")

        # deal with tasks, note that we rename the task to dataset-task
        for task_folder in dataset_path.iterdir():
            if task_folder.is_dir() and task_folder.name != "data":
                task_name = task_folder.name
                target_task_folder = output_path / f"{dataset_name}-{task_name}"
                shutil.copytree(task_folder, target_task_folder)
