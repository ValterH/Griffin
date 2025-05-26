# convert relbench to dbinfer

# 1. load relbench dataset
# 2. convert to dbinfer dataset
# 3. save dbinfer dataset

from relbench.datasets import get_dataset
from relbench.base.task_base import TaskType
from relbench.tasks import get_task_names, get_task
import sys
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from dataset_meta import (
    DBBColumnDType,
    DBBTableDataFormat,
    DBBTableSchema,
    DBBTaskType,
    DBBTaskEvalMetric,
    DBBColumnSchema,
    DBBTaskMeta,
    DBBRDBDatasetMeta,
)
from yaml_utils import save_pyd


def generate_column_schema(column, table):
    dtype = None
    # if the column is table.pkey_col, then it is a primary key
    if column == table.pkey_col:
        dtype = DBBColumnDType.primary_key
    # if the column is table.time_col, then it is a timestamp
    elif column == table.time_col:
        dtype = DBBColumnDType.datetime_t
    elif column in table.fkey_col_to_pkey_table:
        dtype = DBBColumnDType.foreign_key
        link_to = f"{table.fkey_col_to_pkey_table[column]}.{column}"
    elif table.df[column].dtype == float:
        dtype = DBBColumnDType.float_t
    elif (
        table.df[column].dtype == int
        or table.df[column].dtype == np.int32
        or table.df[column].dtype == np.int64
        or table.df[column].dtype == bool
    ):
        dtype = DBBColumnDType.category_t
    elif table.df[column].dtype == object:
        # First get the number of unique values
        n_unique = table.df[column].nunique()
        if n_unique < 10:
            dtype = DBBColumnDType.text_t
        else:
            dtype = DBBColumnDType.category_t
    else:
        # sample 10 rows
        sample = table.df[column].sample(10)
        print(column, sample)
        raise ValueError(f"Unknown column type: {column}")
    column_schema = DBBColumnSchema(name=column, dtype=dtype)
    if dtype == DBBColumnDType.foreign_key:
        column_schema.link_to = link_to
    return column_schema


def generate_column_task_schema(column, table, task):
    dtype = None
    # if the column is table.pkey_col, then it is a primary key
    if column == table.pkey_col:
        dtype = DBBColumnDType.primary_key
    # if the column is table.time_col, then it is a timestamp
    elif column == table.time_col:
        dtype = DBBColumnDType.datetime_t
    elif column == task.entity_col:
        dtype = DBBColumnDType.primary_key
    elif column in table.fkey_col_to_pkey_table:
        dtype = DBBColumnDType.foreign_key
        link_to = f"{table.fkey_col_to_pkey_table[column]}.{column}"
    elif table.df[column].dtype == float:
        dtype = DBBColumnDType.float_t
    elif (
        table.df[column].dtype == int
        or table.df[column].dtype == np.int32
        or table.df[column].dtype == np.int64
        or table.df[column].dtype == bool
    ):
        dtype = DBBColumnDType.category_t
    elif table.df[column].dtype == object:
        dtype = DBBColumnDType.text_t
    else:
        # sample 10 rows
        sample = table.df[column].sample(10)
        print(column, sample)
        raise ValueError(f"Unknown column type: {column}")
    column_schema = DBBColumnSchema(name=column, dtype=dtype)
    if dtype == DBBColumnDType.foreign_key:
        column_schema.link_to = link_to
    return column_schema


def generate_table_schema(table, name):
    column_schemas = []
    for column in table.df.columns:
        column_schema = generate_column_schema(column, table)
        column_schemas.append(column_schema)

    table_schema = DBBTableSchema(
        name=name,
        columns=column_schemas,
        time_column=table.time_col if table.time_col != {} else None,
        format=DBBTableDataFormat.PARQUET,
        source=f"{name}.pqt",
    )
    return table_schema


def generate_task_meta(task, name):
    train_table = task.get_table("train")
    print(task.task_type)
    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        task_type = DBBTaskType.classification
    elif task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        task_type = DBBTaskType.classification
    elif task.task_type == TaskType.REGRESSION:
        task_type = DBBTaskType.regression
    else:
        return None
    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        evaluation_metric = DBBTaskEvalMetric.auroc
    elif task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        evaluation_metric = DBBTaskEvalMetric.accuracy
    elif task.task_type == TaskType.REGRESSION:
        evaluation_metric = DBBTaskEvalMetric.mae
    task_meta = DBBTaskMeta(
        name=name,
        source=f"{name}/{{split}}.pqt",
        format=DBBTableDataFormat.PARQUET,
        columns=[
            generate_column_task_schema(column, train_table, task)
            for column in train_table.df.columns
        ],
        time_column=task.time_col,
        evaluation_metric=evaluation_metric,
        target_column=task.target_col,
        target_table=task.entity_table,
        task_type=task_type,
    )
    return task_meta


def update_foreign_key_links(table_schemas):
    def find_pkey_col(table_name):
        for table_schema in table_schemas:
            if table_schema.name == table_name:
                for column_schema in table_schema.columns:
                    if column_schema.dtype == DBBColumnDType.primary_key:
                        return column_schema.name
        return None
    # For each table, find the foreign key column and update the link_to to the original column name
    for table_schema in table_schemas:
        for column_schema in table_schema.columns:
            if column_schema.dtype == DBBColumnDType.foreign_key:
                target_table_name = column_schema.link_to.split(".")[0]
                target_table_pkey_col = find_pkey_col(target_table_name)
                new_link_to = f"{target_table_name}.{target_table_pkey_col}"
                if new_link_to != column_schema.link_to:
                    print(f"Updating link_to from {column_schema.link_to} to {new_link_to}")
                    column_schema.link_to = new_link_to

    return table_schemas


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-hm")
parser.add_argument("--output_dir", type=str, default="converted_datasets-new")
args = parser.parse_args()
args.output_dir = Path(args.output_dir) / args.dataset
args.output_dir.mkdir(parents=True, exist_ok=True)

dataset = get_dataset(name=args.dataset, download=True)

db = dataset.get_db()
table_schemas = []
for name, table in db.table_dict.items():
    table_schemas.append(generate_table_schema(table, name))

task_metas = []
task_names = get_task_names(args.dataset)
print(task_names)
tasks = [get_task(args.dataset, name) for name in task_names]
for name, task in zip(task_names, tasks):
    task_meta = generate_task_meta(task, name)
    if task_meta is not None:
        task_metas.append(task_meta)

update_foreign_key_links(table_schemas)

dataset_meta = DBBRDBDatasetMeta(
    dataset_name=args.dataset,
    tables=table_schemas,
    tasks=task_metas,
)

save_pyd(dataset_meta, Path(args.output_dir) / "metadata.yaml")

# Save the tables
for table_name, table in db.table_dict.items():
    table.df.to_parquet(Path(args.output_dir) / f"{table_name}.pqt")

# Save the tasks
for task_name, task in zip(task_names, tasks):
    # if the task_name is not in task_metas, then skip
    if task_name not in [task_meta.name for task_meta in task_metas]:
        continue
    os.makedirs(Path(args.output_dir) / task_name, exist_ok=True)
    # Save the train split
    train_table = task.get_table("train")
    train_table.df.to_parquet(Path(args.output_dir) / task_name / "train.pqt")
    # Save the validation split
    val_table = task.get_table("val")
    val_table.df.to_parquet(Path(args.output_dir) / task_name / "validation.pqt")
    # Save the test split
    test_table = task.get_table("test", mask_input_cols=False)
    test_table.df.to_parquet(Path(args.output_dir) / task_name / "test.pqt")
