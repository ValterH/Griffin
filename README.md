# Griffin

This branch provides details for processing raw data to Griffin format.

---

## Table of Contents

- [Griffin](#griffin)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
    - [Prerequisites](#prerequisites)
  - [Step 1: Any-RDB-to-RDB](#step-1-any-rdb-to-rdb)
  - [Step 2: RDB-to-Graph](#step-2-rdb-to-graph)
    - [raw\_for\_griffin](#raw_for_griffin)
    - [generate\_griffin\_feature](#generate_griffin_feature)
    - [construct\_graph](#construct_graph)
  - [Step 3: Graph-to-Huggingface](#step-3-graph-to-huggingface)

---

## Overview

In Griffin, we use datasets from [4DBInfer](https://github.com/awslabs/multi-table-benchmark) and [RelBench](https://github.com/snap-stanford/relbench). To process the raw data to Griffin format, we need following steps:

1. Any-RDB-to-RDB. Convert any RDB to 4DBInfer used RDB format.
2. RDB-to-Graph. Using this repo's preprocess method converting these RDB to graphs.
3. Graph-to-Huggingface. Converting the graphs from DGL format to huggingface format used in Griffin.

### Prerequisites

```bash
bash conda/install-ubuntu-deps.sh
bash conda/create_conda_env.sh
```
> @Valter: For preprocessing I simply created a conda environment with python=3.9 and installed the requirements from conda/requirements.txt:
```bash
conda create -n griffin-preprocessing python=3.9
pip install -r requirements.txt
# optionally install relbench
pip install relbench

```
> @Valter: For preprocessing the relbench datasets simply run the `convert_relbench.sh` script with the dataset name.

## Step 1: Any-RDB-to-RDB

For Relbench datasets, the raw datasets can be downloaded from their github repo. Then it can be converted to 4DBInfer used RDB format by `convert_relbench_to_dbinfer.py`. Note that 4DBInfer baselines require the task table to include all available columns in the target table. Thus if you want to test the 4DBInfer baselines, you need to convert the relbench datasets to 4DBInfer format with the full task table, with script `convert_relbench_to_dbinfer_fulltask.py`.

For 4DBInfer original datasets, we also do a simple modification. Griffin pipeline requires all task tables have the **primary key column**. If the primary key column is not provided, we add the primary key column to the tables. We do this by manual work, and the code are saved at `notebooks/update_*_key.ipynb`.

We also already provide the converted datasets after the Step 1 at [Google Drive](https://drive.google.com/drive/folders/117Wuj5dCvLyPCQrLefBXnBSn40GrEy5N?usp=share_link).

## Step 2: RDB-to-Graph

Preprocess includes 3 steps:

- raw\_for\_griffin
- generate\_griffin\_feature
- construct\_graph

### raw_for_griffin

```bash
python -m tab2graph.main preprocess datasets/$DATASET_NAME transform datasets/$DATASET_NAME-raw -c configs/transform/raw_for_griffin.yaml
```

### generate_griffin_feature

```bash
python -m tab2graph.main preprocess datasets/$DATASET_NAME-raw transform datasets/$DATASET_NAME-single-griffin -c configs/transform/generate_griffin_feature_separate_num.yaml
```

### construct_graph

```bash
python -m tab2graph.main construct-graph datasets/$DATASET_NAME-single-griffin r2n-griffin datasets/$DATASET_NAME-r2n-griffin
```

If you only want to test the 4DBInfer baselines, you can follow the original instructions in [4DBInfer](https://github.com/awslabs/multi-table-benchmark/tree/main/4DBInfer).

## Step 3: Graph-to-Huggingface

Switch to the `main` branch and run:

```bash
python dataconverter.py $SRCPATH $DSTPATH
python dataconverteredge.py $SRCPATH $DSTPATH
python dataconvertertask.py $SRCPATH $DSTPATH
python dataconverterpost.py $SRCPATH