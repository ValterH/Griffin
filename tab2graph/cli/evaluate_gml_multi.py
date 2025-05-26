from pathlib import Path
import typer
import logging
import wandb
import os
import numpy as np

import dbinfer_bench as dbb

from ..device import DeviceInfo, get_device_info
from ..solutions import (
    get_gml_solution_class,
    parse_config_from_graph_dataset_multitask,
    get_gml_solution_choice,
)
from .. import yaml_utils

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

GMLSolutionChoice = get_gml_solution_choice()


def evaluate_gml_multi(
    dataset_path : str = typer.Argument(
        ...,
        help=("Path to the dataset folder or one of the built-in datasets. "
              "Use the list-builtin command to list all the built-in datasets.")
    ),
    task_name_list : str = typer.Argument(
        ...,
        help=("Name of the task to fit the solution.")
    ),
    solution_name : GMLSolutionChoice = typer.Argument(
        ...,
        help="Solution name"
    ),
    checkpoint_path : str = typer.Argument(
        ...,
        help="Path to the workspace. "
        " The workspace should contain data_config.yaml, solution_config.yaml, and model.pt."
    ),
):
    config_path = checkpoint_path + "solution_config.yaml"
    data_config_path = checkpoint_path + "data_config.yaml"

    solution_class = get_gml_solution_class(solution_name.value)

    logger.info(f"Load solution configuration file: {config_path}.")
    solution_config = yaml_utils.load_pyd(solution_class.config_class, config_path)
    logger.debug(f"Solution config:\n{solution_config.json()}")

    logger.info("Loading data ...")
    dataset = dbb.load_graph_data(dataset_path)

    task_name_list = task_name_list.split(',')
    data_config = parse_config_from_graph_dataset_multitask(dataset, task_name_list)
    preserve_data_config = yaml_utils.load_pyd(type(data_config), data_config_path)
    # NOTE: We don't need to assert this because the data config is not used in the evaluation.
    # assert data_config == preserve_data_config, "Preserved data config is different from the original one."
    logger.debug(f"Data config:\n{data_config.json()}")

    logger.info("Creating solution ...")
    # use preserve_data_config instead of data_config,because it includes more tasks
    solution = solution_class(solution_config, preserve_data_config)
    device = get_device_info()

    def _invoke_test(solution, run_ckpt_path : Path, task_name : str, device : DeviceInfo):
        solution.load_from_checkpoint(run_ckpt_path)
        test_metric_dict = solution.evaluate_multi_task(
            dataset.graph_tasks[task_name].test_set,
            dataset.graph,
            dataset.feature,
            device,
            task_name,
        )
        return test_metric_dict

    logger.info("Testing ...")
    for task_name in task_name_list:
        test_metric_dict = _invoke_test(solution, checkpoint_path, task_name, device)
        logger.info(f"Test metric in {task_name}: {test_metric_dict:.4f}")
