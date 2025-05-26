from enum import Enum
from pathlib import Path
import pandas as pd
import numpy as np
from uuid import UUID, uuid5

def degree_info(graph):
    deg = [
        graph.in_degrees(etype=etype)
        for etype in graph.canonical_etypes
    ]
    deg_quantile = [
        np.quantile(d, [0.2, 0.4, 0.6, 0.8, 1.0])
        for d in deg
    ]
    df = pd.DataFrame(
        {
            et : qt
            for (_, et, _), qt in zip(graph.canonical_etypes, deg_quantile)
        },
        index=["20%", "40%", "60%", "80%", "100%"]
    )
    return df.transpose()

def get_sweep_id(checkpoint_path, project_name, sweep_name, sweep_config, attach):
    import wandb

    sweep_file = Path(checkpoint_path) / f'{project_name}-{sweep_name}'

    if attach:
        if not sweep_file.exists():
            raise FileNotFoundError(
                f"Sweep name {sweep_name} not found under checkpoint path {checkpoint_path}"
            )
        sweep_id = sweep_file.read_text()
    else:
        sweep_id = wandb.sweep(
            sweep=sweep_config,
            project=project_name,
        )
        sweep_file.write_text(sweep_id)
    return sweep_id

def get_project_name(model_name, dataset_path, task_name):
    dataset_name = dataset_path.replace('/', '_')
    return f'T2G-sweep-{model_name}-{dataset_name}-{task_name}'

TAB2GRAPH_NAMESPACE_UUID = UUID('42138ced-2d20-4b0c-a4ab-b7b1646a7ba2')

def generate_uuid(*args, **kwargs):
    """
    Generate a UUID for the list of arguments and keyword arguments.
    """
    data = str([str(arg) for arg in args])
    data += str({str(key): str(value) for key, value in kwargs.items()})

    return uuid5(TAB2GRAPH_NAMESPACE_UUID, data)
