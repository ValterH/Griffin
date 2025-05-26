from pathlib import Path
import typer
import logging
import wandb
import os
import numpy as np
from typing import Optional

import dbinfer_bench as dbb

from ..device import DeviceInfo
from ..solutions import (
    get_gml_solution_class,
    parse_config_from_graph_dataset_multitask,
    get_gml_solution_choice,
)
from .. import yaml_utils
from .fit_utils import _fit_main_multitask

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

GMLSolutionChoice = get_gml_solution_choice()

def fit_gml_multi(
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
    test_task_name_list: str = typer.Option(
        None,
        "--test_task_name_list", "-t",
        help="Name of the task to test the solution."
    ),
    config_path : Path = typer.Option(
        None,
        "--config_path", "-c",
        help="Solution configuration path. Use default if not specified."
    ),
    checkpoint_path : str = typer.Option(
        None,
        "--checkpoint_path", "-p",
        help="Checkpoint path."
    ),
    enable_wandb : bool = typer.Option(
        True,
        "--enable-wandb/--disable-wandb",
        help="Enable Weight&Bias for logging."
    ),
    num_runs : int = typer.Option(
        1,
        "--num-runs", "-n",
        help="Number of runs."
    ),
    reload_checkpoint : bool = typer.Option(
        False,
        "--reload_checkpoint", "-r",
        help="Whether to reload the checkpoint."
    ),
    provided_best_val_metric : float = typer.Option(
        float('-inf'),
        "--provided_best_val_metric",
        help="The provided best validation metric."
    ),
    world_size : int = typer.Option(
        1,
        "--world-size", "-w",
        help="Number of GPUs."
    ),
    port : int = typer.Option(
        None,
        "--port", "-P",
        help="Port for the distributed training."
    )
):
    solution_class = get_gml_solution_class(solution_name.value)
    if config_path is None:
        logger.info("No solution configuration file provided. Use default configuration.")
        solution_config = solution_class.config_class()
    else:
        logger.info(f"Load solution configuration file: {config_path}.")
        solution_config = yaml_utils.load_pyd(solution_class.config_class, config_path)

    logger.debug(f"Solution config:\n{solution_config.json()}")

    logger.info("Loading data ...")
    dataset = dbb.load_graph_data(dataset_path)

    task_name_list = task_name_list.split(',')
    data_config = parse_config_from_graph_dataset_multitask(dataset, task_name_list)
    logger.debug(f"Data config:\n{data_config.json()}")
    test_task_name_list = test_task_name_list.split(',') if test_task_name_list is not None else []

    def _invoke_fit(solution, run_ckpt_path : Path, device : DeviceInfo, wandb_run_id : Optional[str] = None):
        if world_size > 1:
            summary = solution.run(
                dataset,
                task_name_list,
                run_ckpt_path,
                device,
                world_size,
                enable_wandb,
                port,
                wandb_run_id
            )
        else:
            summary = solution.fit_multi_task_combined(dataset, task_name_list, test_task_name_list, run_ckpt_path, device)
        return summary

    def _invoke_test(solution, run_ckpt_path : Path, task_name : str, device : DeviceInfo):
        solution.load_from_checkpoint(run_ckpt_path)
        val_metric = solution.evaluate_multi_task(
            dataset.graph_tasks[task_name].validation_set,
            dataset.graph,
            dataset.feature,
            device,
            task_name,
        )
        test_metric = solution.evaluate_multi_task(
            dataset.graph_tasks[task_name].test_set,
            dataset.graph,
            dataset.feature,
            device,
            task_name,
        )
        return val_metric, test_metric

    _fit_main_multitask(
        solution_class,
        dataset,
        task_name_list,
        test_task_name_list,
        data_config,
        solution_config,
        checkpoint_path,
        enable_wandb,
        num_runs,
        reload_checkpoint,
        provided_best_val_metric,
        _invoke_fit,
        _invoke_test
    )
