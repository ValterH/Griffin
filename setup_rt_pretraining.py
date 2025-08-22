import yaml

with open("task_names.yaml", "r") as f:
    task_names = yaml.safe_load(f)

with open("datasets/relfm/metatask.yaml", "r") as f:
    meta_task = yaml.safe_load(f)

all_relbench_tasks = list(meta_task.keys())

held_out_relbench_datasets = [
    "rel-amazon",
    "rel-hm",
    "rel-stack",
]

task_names["relbench"] = all_relbench_tasks

for dataset in held_out_relbench_datasets:
    other_tasks = [task for task in all_relbench_tasks if dataset not in task]
    task_names[f"{dataset}-heldout"] = other_tasks
    print(f"{dataset}-heldout: {other_tasks}")

with open("task_names.yaml", "w") as f:
    yaml.dump(task_names, f, indent=2, default_flow_style=False)