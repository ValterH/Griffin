#!/bin/bash

DATASET=$1

SRCPATH="datasets/728/$DATASET-r2n-griffin"
DSTPATH="datasets/728/relbench/$DATASET"

python dataconverter.py $SRCPATH $DSTPATH
python dataconverteredge.py $SRCPATH $DSTPATH --model-dim=728
python dataconvertertask.py $SRCPATH $DSTPATH
python dataconverterpost.py $SRCPATH --model-dim=728
# python merge_relbench_dataset.py --dataset_name $DATASET --dst_path "datasets/joint-v65"