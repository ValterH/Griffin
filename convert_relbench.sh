#!/bin/bash

DATASET=$1

# requires relbench
python convert_relbench_to_dbinfer_fulltask.py --dataset=$DATASET --output_dir=datasets/728 --add_task_tables

python -m tab2graph.main preprocess datasets/$DATASET transform datasets/$DATASET-raw -c configs/transform/raw_for_griffin.yaml

python -m tab2graph.main preprocess datasets/$DATASET-raw transform datasets/$DATASET-single-griffin -c configs/transform/generate_griffin_feature_separate_num.yaml

python -m tab2graph.main construct-graph datasets/$DATASET-single-griffin r2n-griffin datasets/$DATASET-r2n-griffin

