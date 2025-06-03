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


def generate_column_schema(column, table, name):
    # A patch on rel-f1 dataset
    if args.dataset == "rel-f1" and name == "races":
        if column == "time":
            table.df[column] = pd.to_timedelta(table.df[column]).dt.total_seconds()
    if args.dataset == "rel-trial" and name == "designs":
        if column == "intervention_model" or column == "masking":
            column_schema = DBBColumnSchema(
                name=column, dtype=DBBColumnDType.category_t
            )
            return column_schema
    if args.dataset == "rel-stack" and name == "users":
        if column == "ProfileImageUrl" or column == "WebsiteUrl":
            return None
    if args.dataset == "rel-trial" and name == "outcome_analyses":
        if (
            column == "ci_upper_limit_raw"
            or column == "ci_lower_limit_raw"
            or column == "p_value_raw"
        ):
            return None
    if args.dataset == "rel-trial" and name == "studies":
        if column == "limitations_and_caveats":
            return None
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
    elif table.df[column].dtype == "datetime64[ns]":
        dtype = DBBColumnDType.datetime_t
    elif (
        table.df[column].dtype == float
        or table.df[column].dtype == np.float32
        or table.df[column].dtype == np.float64
    ):
        dtype = DBBColumnDType.float_t
    elif (
        table.df[column].dtype == int
        or table.df[column].dtype == np.int32
        or table.df[column].dtype == np.int64
        or table.df[column].dtype == bool
    ):
        # Based on relbench avito operation, we treat all integer columns as float, too
        dtype = DBBColumnDType.float_t
    elif table.df[column].dtype == object:
        # First get the number of unique values
        try:
            n_unique = table.df[column].nunique()
            if n_unique < 4:
                dtype = DBBColumnDType.category_t
            else:
                dtype = DBBColumnDType.text_t
        except TypeError:
            # Handle unhashable types (e.g., numpy arrays)
            # Treat as text since we can't count unique values
            dtype = DBBColumnDType.text_t
    else:
        # sample 10 rows
        sample = table.df[column].sample(10)
        print(table.df[column].dtype)
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
        # Skip for "Unnamed: 0"
        if column == "Unnamed: 0":
            print(f"Skipping column: {column}")
            continue
        column_schema = generate_column_schema(column, table, name)
        if column_schema is None:
            print(f"Skipping column: {column}")
            continue
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
                    print(
                        f"Updating link_to from {column_schema.link_to} to {new_link_to}"
                    )
                    column_schema.link_to = new_link_to

    return table_schemas


def update_task_metas_with_table_schemas(task_metas, table_schemas):
    """Update task metadata to align with finalized table schemas."""
    modified_task_metas = []
    for task_meta in task_metas:
        new_task_meta = task_meta.copy()
        # Find the target table schema
        target_table_name = task_meta.target_table
        target_table_schema = None
        for table_schema in table_schemas:
            if table_schema.name == target_table_name:
                target_table_schema = table_schema
                break

        if target_table_schema is None:
            print(
                f"Warning: Target table {target_table_name} not found in table schemas"
            )
            raise ValueError(
                f"Target table {target_table_name} not found in table schemas"
            )
            # return task_meta

        # Get the correct primary key column from the target table
        target_table_pkey_col = None
        for column_schema in target_table_schema.columns:
            if column_schema.dtype == DBBColumnDType.primary_key:
                target_table_pkey_col = column_schema.name
                break

        if target_table_pkey_col is None:
            print(f"Warning: No primary key found in target table {target_table_name}")
            raise ValueError(
                f"No primary key found in target table {target_table_name}"
            )
            # return task_meta

        # Update task meta columns to use the correct primary key column name
        for column_schema in task_meta.columns:
            if column_schema.dtype == DBBColumnDType.primary_key:
                if column_schema.name != target_table_pkey_col:
                    print(
                        f"Updating task meta primary key from {column_schema.name} to {target_table_pkey_col}"
                    )
                    column_schema.name = target_table_pkey_col
                break

        # We require different tasks cannot have the same target table & target column
        # If they are the same, we rename their target column to be unique
        print("Task meta target column: ", task_meta.target_column)
        print([it.target_column for it in task_metas if it.name != task_meta.name])
        if task_meta.target_column in [
            it.target_column for it in task_metas if it.name != task_meta.name
        ]:
            print(
                f"Warning: Task {task_meta.name} has the same target column as {task_meta.target_column}"
            )
            new_task_meta.target_column = f"{task_meta.name}"
            # also rename the corresponding column in the task table
            for column_schema in new_task_meta.columns:
                if column_schema.name == task_meta.target_column:
                    column_schema.name = f"{task_meta.name}"
                    break

        # Add the entity table's columns to the task meta
        entity_table_schema = [
            table_schema
            for table_schema in table_schemas
            if table_schema.name == task_meta.target_table
        ][0]
        # Do not include the primary key column, foreign key column, or time column
        entity_table_columns = [
            column_schema
            for column_schema in entity_table_schema.columns
            if column_schema.dtype != DBBColumnDType.primary_key
            and column_schema.dtype != DBBColumnDType.foreign_key
            and column_schema.dtype != DBBColumnDType.datetime_t
        ]
        new_task_meta.columns.extend(entity_table_columns)

        modified_task_metas.append(new_task_meta)
    return modified_task_metas


def update_task_table(table, task_meta, table_schemas, original_relbench_tasks):
    # Currently, this function is only used for updating the task table's primary key column
    # The pk should be the target table's primary key column
    target_table_name = task_meta.target_table
    target_table_schema = [
        table_schema
        for table_schema in table_schemas
        if table_schema.name == target_table_name
    ][0]
    target_table_pkey_col = [
        column_schema
        for column_schema in target_table_schema.columns
        if column_schema.dtype == DBBColumnDType.primary_key
    ][0].name
    # Since this function is for updating the data of task table
    # Thus we need the table.df to be updated
    original_relbench_task = original_relbench_tasks[task_meta.name]
    current_task_table_pkey_col = original_relbench_task.entity_col
    if current_task_table_pkey_col != target_table_pkey_col:
        print(
            f"Updating task table's primary key column from {current_task_table_pkey_col} to {target_table_pkey_col}"
        )
        table.df = table.df.rename(
            columns={current_task_table_pkey_col: target_table_pkey_col}
        )
    else:
        print(
            f"Task table's primary key column is already {current_task_table_pkey_col}"
        )

    # if the target column is renamed, we also need to rename the corresponding column in the task table
    if task_meta.target_column != original_relbench_task.target_col:
        print(
            f"Updating task table's target column from {original_relbench_task.target_col} to {task_meta.target_column}"
        )
        table.df = table.df.rename(
            columns={original_relbench_task.target_col: task_meta.target_column}
        )

    # If added new columns in task_meta columns, we also need to update the task table
    current_num_columns = len(table.df.columns)
    new_num_columns = len(task_meta.columns)
    if current_num_columns < new_num_columns:
        print(f"Adding {new_num_columns - current_num_columns} columns to task table")
        entity_table = db.table_dict[task_meta.target_table]

        # 1) Check if the primary key is the same order as index for efficient access
        entity_pk_values = entity_table.df[entity_table.pkey_col].values
        entity_index_values = entity_table.df.index.values
        pk_matches_index_order = all(entity_pk_values == entity_index_values)

        if pk_matches_index_order:
            print(
                "Primary key values match index order - using efficient index-based access"
            )
        else:
            print(
                "Primary key values don't match index order - using value-based lookup"
            )

        # 2) Generate the columns with the names from task_meta.columns
        columns_to_add = [
            task_meta.columns[i].name
            for i in range(current_num_columns, new_num_columns)
        ]
        print(f"Columns to add: {columns_to_add}")

        # 3) Generate the rows based on task table's current primary key values
        task_pk_values = table.df[target_table_pkey_col].values

        if pk_matches_index_order:
            # Use efficient index-based access
            new_column_data = entity_table.df.loc[task_pk_values, columns_to_add]
        else:
            # Use value-based lookup by setting primary key as index temporarily
            entity_indexed = entity_table.df.set_index(entity_table.pkey_col)
            new_column_data = entity_indexed.loc[task_pk_values, columns_to_add]

        # Reset index to ensure proper alignment
        new_column_data.reset_index(drop=True, inplace=True)

        # Add the new columns to the task table
        for col in columns_to_add:
            table.df[col] = new_column_data[col].values

        print(f"Successfully added {len(columns_to_add)} columns to task table")

    return table


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-hm")
parser.add_argument("--output_dir", type=str, default="converted_with_full_task")
args = parser.parse_args()
args.output_dir = Path(args.output_dir) / args.dataset
args.output_dir.mkdir(parents=True, exist_ok=True)

dataset = get_dataset(name=args.dataset, download=True)
db = dataset.get_db()

# ===== 1. PROCESS DATA TABLES FIRST =====
print("=== Processing Data Tables ===")
table_schemas = []
for name, table in db.table_dict.items():
    print(f"Processing table: {name}")
    table_schemas.append(generate_table_schema(table, name))

# Update foreign key links to ensure consistency
print("=== Updating Foreign Key Links ===")
table_schemas = update_foreign_key_links(table_schemas)

# Save the data tables
print("=== Saving Data Tables ===")
for table_name, table in db.table_dict.items():
    table.df.to_parquet(Path(args.output_dir) / f"{table_name}.pqt")
    print(f"Saved table: {table_name}")

# ===== 2. PROCESS TASKS BASED ON FINALIZED SCHEMAS =====
print("=== Processing Task Metadata ===")
task_names = get_task_names(args.dataset)
print(f"Task names: {task_names}")
tasks = {name: get_task(args.dataset, name) for name in task_names}

task_metas = []
for name, task in tasks.items():
    print(f"Processing task: {name}")
    task_meta = generate_task_meta(task, name)
    if task_meta is not None:
        task_metas.append(task_meta)
    # if task_meta is not None:
    #     # Update task meta based on finalized table schemas
    #     task_meta = update_task_meta_with_table_schemas(task_meta, table_schemas)
    #     task_metas.append(task_meta)

task_metas = update_task_metas_with_table_schemas(task_metas, table_schemas)

# ===== 3. PROCESS AND SAVE TASK TABLES =====
print("=== Processing Task Tables ===")
for task_name, task in tasks.items():
    # Skip if task not in task_metas
    if task_name not in [task_meta.name for task_meta in task_metas]:
        continue

    task_meta = [task_meta for task_meta in task_metas if task_meta.name == task_name][
        0
    ]
    os.makedirs(Path(args.output_dir) / task_name, exist_ok=True)

    print(f"Processing task table: {task_name}")

    # Save the train split
    train_table = task.get_table("train")
    train_table = update_task_table(train_table, task_meta, table_schemas, tasks)
    train_table.df.to_parquet(Path(args.output_dir) / task_name / "train.pqt")

    # Save the validation split
    val_table = task.get_table("val")
    val_table = update_task_table(val_table, task_meta, table_schemas, tasks)
    val_table.df.to_parquet(Path(args.output_dir) / task_name / "validation.pqt")

    # Save the test split
    test_table = task.get_table("test", mask_input_cols=False)
    test_table = update_task_table(test_table, task_meta, table_schemas, tasks)
    test_table.df.to_parquet(Path(args.output_dir) / task_name / "test.pqt")

    print(f"Saved task table: {task_name}")

# ===== 4. CREATE FINAL DATASET METADATA =====
print("=== Creating Dataset Metadata ===")
dataset_meta = DBBRDBDatasetMeta(
    dataset_name=args.dataset,
    tables=table_schemas,
    tasks=task_metas,
)

save_pyd(dataset_meta, Path(args.output_dir) / "metadata.yaml")
print("=== Conversion Complete ===")
