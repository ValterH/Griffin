from typing import Tuple, Dict, Optional, List, Any, Callable, Union
from pathlib import Path
import json
import pydantic
import logging
import wandb
import os
import numpy as np
import pandas as pd
import dbinfer_bench as dbb

from ..device import get_device_info
from .. import yaml_utils
from .utils import generate_uuid

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


class TableProposal:
    columns: Dict[str, np.ndarray]
    metadata: dbb.DBBTableSchema
    already_used_label_columns: List[str]

    def __init__(
        self,
        columns: Dict[str, np.ndarray],
        metadata: dbb.DBBTableSchema,
        already_used_label_columns: List[str],
    ):
        self.columns = columns
        self.metadata = metadata
        self.already_used_label_columns = already_used_label_columns

    def __repr__(self):
        return (
            f"TableProposal(columns={list(self.columns.keys())}, "
            f"metadata={self.metadata}, "
            f"already_used_label_columns={self.already_used_label_columns})"
        )

    def generate_tasks(self) -> List[dbb.DBBRDBTaskCreator]:
        # Iterate columns satisfying the following conditions:
        # 1. The column has not been used as a label column in any task.
        # 2. The column is not the primary key or foreign key.
        # 3. The column is not the time column.
        candidate_columns = {}
        for column_schema in self.metadata.columns:
            if (
                column_schema.dtype
                not in [
                    dbb.DBBColumnDType.primary_key,
                    dbb.DBBColumnDType.foreign_key,
                ]
                and column_schema.name not in self.already_used_label_columns
                and column_schema.name != self.metadata.time_column
            ):
                # For category columns, we only keep types with more than 1 unique values.
                if column_schema.dtype == dbb.DBBColumnDType.category_t:
                    if (
                        len(pd.unique(self.columns[column_schema.name])) > 1
                        and len(pd.unique(self.columns[column_schema.name])) < 10
                    ):
                        candidate_columns[column_schema.name] = self.columns[
                            column_schema.name
                        ]
                elif column_schema.dtype == dbb.DBBColumnDType.float_t:
                    # For numerical columns, we do not have such restriction.
                    candidate_columns[column_schema.name] = self.columns[
                        column_schema.name
                    ]
                else:
                    # We do not consider other types for now.
                    pass

        tasks = []
        for i, (column_name, column_data) in enumerate(candidate_columns.items()):
            task = self.generate_task(column_name, random_state=i + 42)
            if task is not None:
                tasks.append(task)
        return tasks

    def generate_tasks_with_num(
        self, num_classification: int, num_regression: int
    ) -> List[dbb.DBBRDBTaskCreator]:
        # Generate tasks with the given number of classification and regression tasks.
        current_num_classification = 0
        current_num_regression = 0
        candidate_columns = {}
        for column_schema in self.metadata.columns:
            if (
                column_schema.dtype
                not in [
                    dbb.DBBColumnDType.primary_key,
                    dbb.DBBColumnDType.foreign_key,
                ]
                and column_schema.name not in self.already_used_label_columns
                and column_schema.name != self.metadata.time_column
                and not column_schema.name.startswith("Unnamed")
            ):
                if column_schema.dtype == dbb.DBBColumnDType.category_t:
                    if (
                        len(pd.unique(self.columns[column_schema.name])) > 1
                        and len(pd.unique(self.columns[column_schema.name])) < 10
                        and current_num_classification < num_classification
                    ):
                        candidate_columns[column_schema.name] = self.columns[
                            column_schema.name
                        ]
                        current_num_classification += 1
                elif column_schema.dtype == dbb.DBBColumnDType.float_t:
                    if current_num_regression < num_regression:
                        candidate_columns[column_schema.name] = self.columns[
                            column_schema.name
                        ]
                        current_num_regression += 1
                else:
                    # We do not consider other types for now.
                    pass

        tasks = []
        for i, (column_name, column_data) in enumerate(candidate_columns.items()):
            # All tasks share the same split.
            task = self.generate_task(column_name, random_state=42)
            if task is not None:
                tasks.append(task)
        return tasks

    def generate_tasks_with_given_task_file(
        self, given_task_file: str
    ) -> List[dbb.DBBRDBTaskCreator]:
        # Read the given task file.
        with open(given_task_file, "r") as f:
            given_tasks = json.load(f)
        # Find the target column name and task type.
        target_column_name = given_tasks["target_name"]
        task_type = given_tasks["task"]
        task = self.generate_task(
            target_column_name, task_type=task_type, random_state=42
        )
        return [task]

    def generate_task(
        self, target_column_name: str, task_type: str = None, random_state: int = 42
    ) -> dbb.DBBRDBTaskCreator:
        # First check the primary key and foreign key columns.
        pk_columns = [
            column_schema.name
            for column_schema in self.metadata.columns
            if column_schema.dtype == dbb.DBBColumnDType.primary_key
        ]
        fk_columns = [
            column_schema.name
            for column_schema in self.metadata.columns
            if column_schema.dtype == dbb.DBBColumnDType.foreign_key
        ]
        if len(pk_columns) != 1 and len(fk_columns) != 1:
            return None
        # Set up the task creator.
        target_column_dtype = [
            column_schema.dtype
            for column_schema in self.metadata.columns
            if column_schema.name == target_column_name
        ][0]
        task_ctor = dbb.DBBRDBTaskCreator(f"{self.metadata.name}-{target_column_name}")
        (
            task_ctor.set_task_type(dbb.DBBTaskType.classification)
            if target_column_dtype == dbb.DBBColumnDType.category_t
            else task_ctor.set_task_type(dbb.DBBTaskType.regression)
        )
        if task_type is not None:
            assert (
                task_ctor.task_fields["task_type"] == dbb.DBBTaskType.classification
                if task_type == "classification"
                else True
            )
            assert (
                task_ctor.task_fields["task_type"] == dbb.DBBTaskType.regression
                if task_type == "regression"
                else True
            )
        if task_ctor.task_fields["task_type"] == dbb.DBBTaskType.classification:
            task_ctor.add_task_field(
                "num_classes", len(pd.unique(self.columns[target_column_name]))
            )
        task_ctor.set_evaluation_metric(
            dbb.DBBTaskEvalMetric.auroc
            if task_ctor.task_fields["task_type"] == dbb.DBBTaskType.classification
            and task_ctor.task_fields["num_classes"] == 2
            else (
                dbb.DBBTaskEvalMetric.accuracy
                if task_ctor.task_fields["task_type"] == dbb.DBBTaskType.classification
                and task_ctor.task_fields["num_classes"] > 2
                else dbb.DBBTaskEvalMetric.rmse
            )
        )
        task_ctor.set_target_table(self.metadata.name)
        task_ctor.set_target_column(target_column_name)
        task_ctor.set_key_prediction_label_column("label")
        task_ctor.set_key_prediction_query_idx_column("query_idx")
        # task_ctor.set_time_column(None)

        # Add the task data.
        # Random split to train/val/test.
        n = len(self.columns[target_column_name])
        ids = np.random.RandomState(random_state).permutation(np.arange(n))
        train_idx, val_idx, test_idx = np.split(ids, [int(n * 0.8), int(n * 0.9)])
        for column_schema in self.metadata.columns:
            column_name = column_schema.name
            # only add the task data for the target column and the primary key column.
            if column_name not in [target_column_name, pk_columns[0]]:
                continue
            task_ctor.add_task_data(
                train_data=self.columns[column_name][train_idx],
                validation_data=self.columns[column_name][val_idx],
                test_data=self.columns[column_name][test_idx],
                **column_schema.dict(),
            )

        return task_ctor


def find_pk(task: dbb.DBBRDBTask) -> Tuple[bool, Optional[str]]:
    columns = task.metadata.columns
    for column in columns:
        if column.dtype == dbb.DBBColumnDType.primary_key:
            return True, column.name
    return False, None


def mask_test_split(
    table_proposal: TableProposal, task: dbb.DBBRDBTask
) -> TableProposal:
    _, pk_column = find_pk(task)
    assert pk_column is not None
    # Mask the primary-key rows of the test split.
    test_mask = np.isin(table_proposal.columns[pk_column], task.test_set[pk_column])
    # For all the columns, keep the values not in the test split.
    for k, v in table_proposal.columns.items():
        table_proposal.columns[k] = v[~test_mask]
    table_proposal.already_used_label_columns.append(task.metadata.target_column)
    return table_proposal


def construct_data_table(
    task: dbb.DBBRDBTask, table_proposal: TableProposal
) -> TableProposal:
    # Re-construct the data table by using the task.train_set & task.validation_set.
    updated_columns = {}
    left_columns = []
    for column_schema in task.metadata.columns:
        column_name = column_schema.name
        for task_data_column_schema in task.metadata.columns:
            if task_data_column_schema.name == column_name:
                updated_columns[column_name] = np.concatenate(
                    [task.train_set[column_name], task.validation_set[column_name]]
                )
                left_columns.append(column_name)
                break
    table_proposal.metadata.columns = [
        column_schema
        for column_schema in table_proposal.metadata.columns
        if column_schema.name in left_columns
    ]
    table_proposal.columns = updated_columns
    table_proposal.already_used_label_columns.append(task.metadata.target_column)
    return table_proposal


def construct_tasks(table_proposal: TableProposal) -> List[dbb.DBBRDBTaskCreator]:
    # Iterate over all the columns to propose a new task for each column.
    tasks = table_proposal.generate_tasks()
    return tasks


def regenerate_dataset(
    dataset: dbb.DBBRDBDataset, tasks: List[dbb.DBBRDBTaskCreator], output_path: str
) -> None:
    dataset_creator = dbb.DBBRDBDatasetCreator(dataset.metadata.dataset_name)
    for table_metadata in dataset.metadata.tables:
        table_name = table_metadata.name
        dataset_creator.add_table(table_name)
        time_col = table_metadata.time_column
        for column_schema in table_metadata.columns:
            col_name = column_schema.name
            # Generate the left column metadata beyond name and dtype.
            additional_meta_dict = column_schema.dict()
            additional_meta_dict.pop("name")
            additional_meta_dict.pop("dtype")
            dataset_creator.add_column(
                table_name=table_name,
                column_name=col_name,
                data=dataset.tables[table_name][col_name],
                dtype=column_schema.dtype,
                **additional_meta_dict,
            )
        if time_col is not None:
            dataset_creator.set_time_column(table_name, time_col)
    # TODO(yamboo): Check do we need to add column groups.
    # if dataset.column_groups is not None:
    #     for col_group in dataset.column_groups:
    #         dataset_creator.add_column_group(col_group)

    # Add the original tasks.
    for original_task in dataset.tasks:
        task_ctor = dbb.DBBRDBTaskCreator(original_task.metadata.name)
        task_ctor.task_fields = original_task.metadata.dict()
        # Update task_fields columns.
        # Convert from list to dict
        task_ctor.task_fields["columns"] = {}
        for column_schema in original_task.metadata.columns:
            task_ctor.task_fields["columns"][column_schema.name] = column_schema.dict()
        # Add task data.
        for column_schema in original_task.metadata.columns:
            column_name = column_schema.name
            train_data, validation_data, test_data = (
                original_task.train_set[column_name],
                original_task.validation_set[column_name],
                original_task.test_set[column_name],
            )
            task_ctor.add_task_data(
                train_data=train_data,
                validation_data=validation_data,
                test_data=test_data,
                **column_schema.dict(),
            )
        if original_task.metadata.task_type == dbb.DBBTaskType.classification:
            pass
        elif original_task.metadata.task_type == dbb.DBBTaskType.retrieval:
            # Copy extra columns needed by retrieval task.
            for col_name in [
                original_task.metadata.key_prediction_label_column,
                original_task.metadata.key_prediction_query_idx_column,
            ]:
                task_ctor.add_task_data(
                    col_name,
                    None,
                    original_task.validation_set[col_name],
                    original_task.test_set[col_name],
                    dtype=None,
                )
        dataset_creator.add_task(task_ctor)
    # Add the new tasks.
    for task in tasks:
        dataset_creator.add_task(task)
    dataset_creator.done(output_path)
