import typer
import logging
import os
import copy
import numpy as np
import shutil

import dbinfer_bench as dbb

from .. import yaml_utils
from typing import Dict, List
from pathlib import Path
from .generate_single_task_utils import (
    TableProposal,
    find_pk,
    mask_test_split,
    construct_data_table,
    construct_tasks,
    regenerate_dataset,
)

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


def generate_single_task(
    dataset_path: str = typer.Argument(
        ...,
        help=(
            "Path to the dataset folder or one of the built-in datasets. "
            "Use the list-builtin command to list all the built-in datasets."
        ),
    ),
    output_path: str = typer.Argument(
        ..., help="Output path for the preprocessed dataset."
    ),
    num_classification: int = typer.Option(
        0,
        "--num-classification",
        help="Number of classification tasks to generate.",
    ),
    num_regression: int = typer.Option(
        0,
        "--num-regression",
        help="Number of regression tasks to generate.",
    ),
    with_given_task_file: str = typer.Option(
        None,
        "--task-file",
        help="Path to the given task file. If provided, the task file will be used to generate tasks for each table.",
    ),
):
    logger.info(f"Loading dataset from {dataset_path}.")
    dataset = dbb.load_rdb_data(dataset_path)

    # Step 1: Find other tasks in the dataset.
    # For those tasks where the targeted label is in the dataset, we query their primary-key columns.
    # We mask the primary-key rows of the test split, and query original data as pre-training data.
    # If the primary-key column is missing, we cannot know which rows are masked.
    # Therefore, we do not use the original table as pre-training data.
    # Instead, we collect training data from the task table.

    table_proposals = {
        table_metadata.name: TableProposal(
            columns=dataset.tables[table_metadata.name].copy(),
            metadata=table_metadata,
            already_used_label_columns=[],
        )
        for table_metadata in dataset.metadata.tables
    }

    tasks_with_pk = []
    tasks_without_pk = []
    for task in dataset.tasks:
        if find_pk(task)[0]:
            tasks_with_pk.append(task)
        else:
            tasks_without_pk.append(task)
    # Verify that no table is used as target table for multiple tasks without primary key.
    for task in tasks_without_pk:
        assert task.metadata.target_table not in [
            t.metadata.target_table for t in tasks_without_pk if t != task
        ]

    logger.info("Starting to preprocess the dataset based on the existing tasks.")

    for task in tasks_with_pk:
        # Mask the primary-key rows of the test split.
        table_proposals[task.metadata.target_table] = mask_test_split(
            table_proposals[task.metadata.target_table], task
        )

    for task in tasks_without_pk:
        # Refactor the task to use the pre-training data.
        table_proposals[task.metadata.target_table] = construct_data_table(
            task, table_proposals[task.metadata.target_table]
        )

    logger.info("Creating new tasks for each table.")
    new_tasks = []
    for table_name, table_proposal in table_proposals.items():
        # Construct new tasks for each table.
        if table_proposal is not None:
            if with_given_task_file:
                with_given_task_file = Path(dataset_path) / with_given_task_file
                constructed_tasks = table_proposal.generate_tasks_with_given_task_file(
                    with_given_task_file
                )
            elif num_classification > 0 or num_regression > 0:
                constructed_tasks = table_proposal.generate_tasks_with_num(
                    num_classification, num_regression
                )
            else:
                constructed_tasks = table_proposal.generate_tasks()
            new_tasks.extend(constructed_tasks)

    logger.info(f"Created {len(new_tasks)} new tasks.")
    if len(new_tasks) == 0:
        logger.warning("No new tasks created. Please check the dataset and the number of tasks to generate.")
        return
    for task in new_tasks:
        logger.info(f"Task: {task.task_fields['name']}")

    logger.info("Saving the preprocessed dataset.")
    # Force to remove the output path if it exists.
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    regenerate_dataset(dataset, new_tasks, output_path)

    logger.info(f"Preprocessed dataset saved to {output_path}.")
