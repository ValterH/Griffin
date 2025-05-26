from typing import Tuple, Dict, Optional, List, Any, Callable, Union
from pathlib import Path
import pydantic
import logging
import wandb
import os
import numpy as np

from ..device import get_device_info
from .. import yaml_utils
from .utils import generate_uuid

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')


def _fit_main(
    solution_class,
    dataset,
    task_name : str,
    data_config : pydantic.BaseModel,
    solution_config : Optional[pydantic.BaseModel], # None for loading from WandB (during sweeps)
    checkpoint_path : str,
    enable_wandb : bool,
    num_runs : int,
    invoke_fit : Callable,
    invoke_test : Callable,
):
    device = get_device_info()
    if checkpoint_path is None:
        checkpoint_path = os.path.join(
            os.path.curdir,
            '_checkpoints',
            str(generate_uuid(
                solution_class,
                dataset,
                data_config,
                solution_config,
            ))
        )
        logger.info(f'Checkpoint path not specified, using {checkpoint_path}.')
    checkpoint_path = Path(checkpoint_path)

    if num_runs > 1 and enable_wandb:
        # Create run groups in wandb.
        os.environ["WANDB_RUN_GROUP"] = "exp-" + wandb.util.generate_id()

    run_metrics = []
    train_metrics = []
    for run in range(num_runs):
        if solution_config is not None:
            # Solution config given outside, not from WandB
            wandb_config = {k : v for k, v in solution_config.__dict__.items()}
            wandb_config["solution"] = solution_class.name
            wandb_config["dataset"] = dataset.dataset_name
            wandb_config["task"] = task_name
            wandb.init(
                project="Tab2graph",
                mode="online" if enable_wandb else "disabled",
                config=wandb_config
            )
            wandb.define_metric("val_metric", summary="max")
            wandb_run_id = wandb.run.id
        else:
            # Solution config is from WandB (can only be loaded after wandb.init())
            wandb.init(
                project="Tab2graph",
                mode="online" if enable_wandb else "disabled",
            )
            wandb.define_metric("val_metric", summary="max")
            param_config = dict(wandb.config)
            solution_config = solution_class.config_class.parse_obj(param_config)

        run_checkpoint_path = checkpoint_path / f"{solution_class.name}-run-{run}"
        run_checkpoint_path.mkdir(parents=True, exist_ok=True)

        logger.info("Creating solution ...")
        solution = solution_class(solution_config, data_config)

        logger.info("Fitting ...")
        summary = invoke_fit(solution, run_checkpoint_path, device, wandb_run_id)
        run_metrics.append(summary.val_metric)
        train_metric = summary.train_metric
        train_metrics.append(train_metric)

        logger.info("Testing ...")
        val_metric, test_metric = invoke_test(solution, run_checkpoint_path, device)
        logger.info(f"Train metric: {train_metric:.4f}")
        logger.info(f"Validation metric: {val_metric:.4f}")
        logger.info(f"Test metric: {test_metric:.4f}")
        wandb.log({'test_metric': test_metric})

    if num_runs > 1:
        # Re-test on the best run.
        best_metrics = max(run_metrics)
        best_run = np.argmax(run_metrics)
        train_metric = train_metrics[best_run]
        logger.info(f"""Summary of {num_runs} runs:
    best metric: {best_metrics:.4f}
    best run: {best_run}
    average: {np.mean(run_metrics):.4f}
    std: {np.std(run_metrics):.4f}
""")
        best_checkpoint_path = checkpoint_path / f"{solution_class.name}-run-{best_run}"
        logger.info("Testing using the best checkpoint ...")
        val_metric, test_metric = invoke_test(solution, best_checkpoint_path, device)
        logger.info(f"Train metric: {train_metric:.4f}")
        logger.info(f"Validation metric: {val_metric:.4f}")
        logger.info(f"Test metric: {test_metric:.4f}")

    return train_metric, val_metric, test_metric


def _fit_main_multitask(
    solution_class,
    dataset,
    task_name_list : List[str],
    test_task_name_list : List[str],
    data_config : pydantic.BaseModel,
    solution_config : Optional[pydantic.BaseModel],  # None for loading from WandB (during sweeps)
    checkpoint_path : str,
    enable_wandb : bool,
    num_runs : int,
    reload_checkpoint : bool,
    provided_best_val_metric : float,
    invoke_fit : Callable,
    invoke_test : Callable,
):
    device = get_device_info()
    if checkpoint_path is None:
        checkpoint_path = os.path.join(
            os.path.curdir,
            '_checkpoints',
            str(generate_uuid(
                solution_class,
                dataset,
                data_config,
                solution_config,
            ))
        )
        logger.info(f'Checkpoint path not specified, using {checkpoint_path}.')
    checkpoint_path = Path(checkpoint_path)

    if num_runs > 1 and enable_wandb:
        # Create run groups in wandb.
        AssertionError("num_runs > 1 not supported yet for multi-task.")
        os.environ["WANDB_RUN_GROUP"] = "exp-" + wandb.util.generate_id()

    run_metrics = []
    train_metrics = []
    for run in range(num_runs):
        if solution_config is not None:
            # Solution config given outside, not from WandB
            wandb_config = {k : v for k, v in solution_config.__dict__.items()}
            wandb_config["solution"] = solution_class.name
            wandb_config["dataset"] = dataset.dataset_name
            wandb_config["task_list"] = task_name_list
            wandb.init(
                project="Tab2graph",
                mode="online" if enable_wandb else "disabled",
                config=wandb_config
            )
            wandb.define_metric("val_metric", summary="max")
            wandb_run_id = wandb.run.id
        else:
            # Solution config is from WandB (can only be loaded after wandb.init())
            wandb.init(
                project="Tab2graph",
                mode="online" if enable_wandb else "disabled",
            )
            wandb.define_metric("val_metric", summary="max")
            param_config = dict(wandb.config)
            solution_config = solution_class.config_class.parse_obj(param_config)

        run_checkpoint_path = checkpoint_path / f"{solution_class.name}-run-{run}"
        run_checkpoint_path.mkdir(parents=True, exist_ok=True)

        logger.info("Creating solution ...")
        solution = solution_class(solution_config, data_config, provided_best_val_metric=provided_best_val_metric)

        if reload_checkpoint:
            logger.info("Reloading checkpoint ...")
            solution.load_from_checkpoint(run_checkpoint_path)

        logger.info("Fitting ...")
        summary = invoke_fit(solution, run_checkpoint_path, device, wandb_run_id)
        # Reimport wandb if multi-gpu
        wandb.init(
            project="Tab2graph",
            mode="online" if enable_wandb else "disabled",
            id=wandb_run_id,
            resume="must"
        )
        run_metrics.append(summary.val_metric)
        train_metric = summary.train_metric
        train_metrics.append(train_metric)

        logger.info("Testing ...")
        if test_task_name_list:
            train_task_name_list = [task_name for task_name in task_name_list if task_name not in test_task_name_list]
            for task_name in train_task_name_list:
                logger.info(f"Task: {task_name}")
                val_metric, test_metric = invoke_test(solution, run_checkpoint_path, task_name, device)
                logger.info(f"Train metric: {train_metric:.4f}")
                logger.info(f"Validation metric: {val_metric:.4f}")
                logger.info(f"Test metric: {test_metric:.4f}")
                wandb.log({f'test_metric/{task_name}': test_metric})
            for task_name in test_task_name_list:
                logger.info(f"Task: {task_name}")
                val_metric_dict, test_metric_dict = invoke_test(solution, run_checkpoint_path, task_name, device)
                logger.info(f"Train metric: {train_metric:.4f}")
                for evaluation_setting in val_metric_dict.keys():
                    val_metric = val_metric_dict[evaluation_setting]
                    test_metric = test_metric_dict[evaluation_setting]
                    logger.info(f"Validation metric in {evaluation_setting}-shot: {val_metric:.4f}")
                    logger.info(f"Test metric in {evaluation_setting}-shot: {test_metric:.4f}")
                    wandb.log({f'test_metric/{task_name}/{evaluation_setting}-shot': test_metric})
        else:
            for task_name in task_name_list:
                logger.info(f"Task: {task_name}")
                val_metric, test_metric = invoke_test(solution, run_checkpoint_path, task_name, device)
                logger.info(f"Train metric: {train_metric:.4f}")
                logger.info(f"Validation metric: {val_metric:.4f}")
                logger.info(f"Test metric: {test_metric:.4f}")
                wandb.log({f'test_metric/{task_name}': test_metric})

    if num_runs > 1:
        AssertionError("num_runs > 1 not supported yet for multi-task.")
        # Re-test on the best run.
        best_metrics = max(run_metrics)
        best_run = np.argmax(run_metrics)
        train_metric = train_metrics[best_run]
        logger.info(f"""Summary of {num_runs} runs:
    best metric: {best_metrics:.4f}
    best run: {best_run}
    average: {np.mean(run_metrics):.4f}
    std: {np.std(run_metrics):.4f}
""")
        best_checkpoint_path = checkpoint_path / f"{solution_class.name}-run-{best_run}"
        logger.info("Testing using the best checkpoint ...")
        val_metric, test_metric = invoke_test(solution, best_checkpoint_path, device)
        logger.info(f"Train metric: {train_metric:.4f}")
        logger.info(f"Validation metric: {val_metric:.4f}")
        logger.info(f"Test metric: {test_metric:.4f}")
