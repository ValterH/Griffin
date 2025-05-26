from pathlib import Path
import typer
import logging
import wandb
import os
import numpy as np
import pandas as pd
import yaml
from enum import Enum

import dbinfer_bench as dbb

from ..device import DeviceInfo
from ..solutions import (
    get_tabml_solution_class,
    parse_config_from_tabular_dataset,
    get_tabml_solution_choice,
    SweepChoice,
)
from .. import yaml_utils
from .fit_utils import _fit_main
from .utils import get_sweep_id, get_project_name, generate_uuid

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

TabMLSolutionChoice = get_tabml_solution_choice()


def tune_tab(
    dataset_path : str = typer.Argument(
        ...,
        help=("Path to the dataset folder or one of the built-in datasets. "
              "Use the list-builtin command to list all the built-in datasets.")
    ),
    task_name : str = typer.Argument(
        ...,
        help=("Name of the task to fit the solution.")
    ),
    solution_name : TabMLSolutionChoice = typer.Argument(
        ...,
        help="Solution name"
    ),
    config_path : str = typer.Argument(
        ...,
        help="Sweep configuration path."
    ),
    sweep_name : str = typer.Argument(
        None,
        help=("An identifier for the sweep. "
              "Can be used to attach to existing sweeps with the -a option.")
    ),
    sweep_method : SweepChoice = typer.Option(
        "random",
        "--sweep_method", "-m",
        help="Sweep (hyper-parameter search) method."
    ),
    sweep_count : int = typer.Option(
        100,
        "--sweep_count", "-c",
        help=("Number of hyper-parameter combinations to try. "
              "Setting it to -1 means exhuasting all the combinations. "
              "However, this may cause the sweep to run forever.")
    ),
    checkpoint_path : str = typer.Option(
        None, 
        "--checkpoint_path", "-p",
        help="Checkpoint path."
    ),
    attach : bool = typer.Option(
        False,
        "--attach", "-a",
        help=("Attach to existing sweep with name identified as SWEEP_NAME. "
              "If false, create a new sweep instead.")
    ),
    # Used by safe_tune.sh for programmatically returning the best config and metric
    return_sweep_path : bool = typer.Option(
        False,
        "--return_sweep_path", "-r",
        help=("Only return the sweep path without running actual sweep.  Implies -a")
    ),
    time_budget : str = typer.Option(
        "1h",
        "--time_budget", "-t",
        help="Time budget per trial (1h)",
    ),
):
    solution_class = get_tabml_solution_class(solution_name.value)
    with open(config_path, 'r') as f:
        param_grid = yaml.safe_load(f)
    param_grid['time_budget'] = {
        'distribution': 'constant',
        'value': pd.to_timedelta(time_budget).total_seconds(),
    }

    logger.info('Sweep config:')
    sweep_config = {
        'method': sweep_method.value,
        'metric': {'name': 'val_metric.max', 'goal': 'maximize'},
        'parameters': param_grid,
    }
    logger.info(sweep_config)

    logger.info("Loading data ...")
    dataset = dbb.load_rdb_data(dataset_path)
    model_name = sweep_config['parameters']['nn_name']['value']

    data_config = parse_config_from_tabular_dataset(dataset, task_name)
    logger.debug(f"Data config:\n{data_config.json()}")
    dataset_name = data_config

    project = get_project_name(model_name, dataset_path, task_name)

    if checkpoint_path is None:
        checkpoint_path = os.path.join(
            os.path.curdir,
            '_checkpoints',
            str(generate_uuid(
                solution_class,
                data_config,
                project,
                sweep_name,
                sweep_config,
            ))
        )
        logger.info(f'Checkpoint path not specified, using {checkpoint_path}.')
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    sweep_id = get_sweep_id(checkpoint_path, project, sweep_name, sweep_config, attach or return_sweep_path)
    if return_sweep_path:
        print(f'{project}/{sweep_id}')
        return

    def _invoke_fit(solution, run_ckpt_path : Path, device : DeviceInfo):
        summary = solution.fit(dataset, task_name, run_ckpt_path, device)
        return summary

    def _invoke_test(solution, run_ckpt_path : Path, device : DeviceInfo):
        solution.load_from_checkpoint(run_ckpt_path)
        val_metric = solution.evaluate(
            dataset.get_task(task_name).validation_set,
            device
        )
        test_metric = solution.evaluate(
            dataset.get_task(task_name).test_set,
            device
        )
        return val_metric, test_metric

    reported_train_metrics = []
    reported_val_metrics = []
    reported_test_metrics = []
    reported_configs = []

    def _fit_one():
        train_metric, val_metric, test_metric = _fit_main(
            solution_class,
            dataset,
            data_config,
            None,
            checkpoint_path,
            True,
            1,
            _invoke_fit,
            _invoke_test
        )
        reported_train_metrics.append(train_metric)
        reported_val_metrics.append(val_metric)
        reported_test_metrics.append(test_metric)
        reported_configs.append(wandb.config)

    if sweep_count < 0:
        sweep_count = None
    wandb.agent(sweep_id, function=_fit_one, count=sweep_count, project=project)

    best_idx = np.argmax(reported_val_metrics)
    logger.info("Best hyperparameter config:")
    logger.info(reported_configs[best_idx])
    logger.info(f"Best train: {reported_train_metrics[best_idx]:.4f}")
    logger.info(f"Best val: {reported_val_metrics[best_idx]:.4f}")
    logger.info(f"Best test: {reported_test_metrics[best_idx]:.4f}")
