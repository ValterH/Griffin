import os
import yaml
import shutil
import argparse


import torch
import datasets as hds

def update_dataset(dataset, dataset_name):
    """
    Rename features in the dataset according to the dataset_name prefixing rules.
    """
    features = list(dataset.features.keys())
    rename_map = {}
    for key in features:
        if key.startswith("head of "):
            new_key = key.replace("head of ", f"head of {dataset_name}-")
            new_key = new_key.replace(":", f":{dataset_name}-")
            rename_map[key] = new_key
        elif key.startswith("tail of"):
            new_key = key.replace("tail of ", f"tail of {dataset_name}-")
            new_key = new_key.replace(":", f":{dataset_name}-")
            rename_map[key] = new_key
    if rename_map:
        dataset = dataset.rename_columns(rename_map)
    return dataset

def copy_subfolders(src, dst, dataset_name):
    """
    Copy subfolders from src to dst, renaming them with dataset_name prefix.
    """
    if not os.path.exists(dst):
        os.makedirs(dst)
    
    for folder in os.listdir(src):
        folder_src = os.path.join(src, folder)
        if os.path.isdir(folder_src):
            folder_dst = os.path.join(dst, f"{dataset_name}-{folder}")
            if os.path.exists(folder_dst):
                print(f"Destination folder {folder_dst} already exists, removing it.")
                shutil.rmtree(folder_dst)
            shutil.copytree(folder_src, folder_dst)
            print(f"Copied {folder_src} to {folder_dst}")

        else:
            print(f"Skipping {folder_src}, not a directory.")

def update_keys(dst_dict, src_dict, dataset_name=None):
    # update all src_dict keys with dataset_name prefix
    old_keys = list(src_dict.keys())
    for key in old_keys:
        new_key = f"{dataset_name}-{key}"
        # update subkeys in the dictionary
        if isinstance(src_dict[key], dict):
            for subkey in src_dict[key]:
                if subkey == "out":
                    items = []
                    for item in src_dict[key][subkey]:
                        item = item.replace("head of ", f"head of {dataset_name}-")
                        item = item.replace(":", f":{dataset_name}-")
                        items.append(item)
                    src_dict[key][subkey] = items
                elif subkey == "in":
                    items = []
                    for item in src_dict[key][subkey]:
                        item = item.replace("tail of ", f"tail of {dataset_name}-")
                        item = item.replace(":", f":{dataset_name}-")
                        items.append(item)
                    src_dict[key][subkey] = items
                elif subkey == "feat":
                    # items = [f"{dataset_name}-{item}" for item in src_dict[key][subkey]]
                    # src_dict[key][subkey] = items
                    pass
                elif subkey == "name" and src_dict[key][subkey]:
                    src_dict[key][subkey] = f"{dataset_name}-{src_dict[key][subkey]}"
                elif subkey == 'seed_type' and src_dict[key][subkey]:
                    src_dict[key][subkey] = f"{dataset_name}-{src_dict[key][subkey]}"
                elif subkey == 'target_type' and src_dict[key][subkey]:
                    src_dict[key][subkey] = f"{dataset_name}-{src_dict[key][subkey]}"
                elif subkey == "extra_feat":
                    assert len(src_dict[key][subkey]) == 0
                elif subkey == "masked_feat":
                    assert len(src_dict[key][subkey]) == 0
                else:
                    assert subkey in [
                        "num", 
                        "is_target", 
                        "hastimestamp", 
                        "metric", 
                        "num_class", 
                        "split",
                        "task_type"
                    ], f"Unexpected subkey {subkey} in {key} of src_dict"

        dst_dict.pop(key, None) # remove the old key if it exists
        dst_dict[new_key] = src_dict.pop(key)
        print(f"Updating key {new_key}.")
            
    return dst_dict

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Merge RelBench dataset into joint-v65")
    argparser.add_argument("--dataset_name", type=str, default="rel-avito", help="Name of the dataset to merge")
    argparser.add_argument("--dst_path", type=str, default="datasets/relfm", help="Destination path for the merged dataset")
    # argparser.add_argument("--model-dim", type=int, default=728, help="Dimension of the model")

    args = argparser.parse_args()
    dataset_name = args.dataset_name
    dataset_path = os.path.join("datasets", "relbench-728", dataset_name)
    destination_path = args.dst_path

    os.makedirs(destination_path, exist_ok=True)
    os.makedirs(os.path.join(destination_path, "edge"), exist_ok=True)
    os.makedirs(os.path.join(destination_path, "task"), exist_ok=True)
    os.makedirs(os.path.join(destination_path, "meta"), exist_ok=True)
    # combine edgenameemb.pt files
    edgenameemb_dataset = torch.load(os.path.join(dataset_path, "edgenameemb.pt"))
    try:
        edgenameemb_dst = torch.load(os.path.join(destination_path, "edgenameemb.pt"))
    except FileNotFoundError:
        edgenameemb_dst = dict()
        edgenameemb_dst["fewshot"] = torch.load("datasets/joint-v65/edgenameemb.pt")["fewshot"]
    for key in edgenameemb_dataset.keys():
        if key.startswith("head of "):
            new_key = key.replace("head of ", f"head of {dataset_name}-")
            new_key = new_key.replace(":", f":{dataset_name}-")
            edgenameemb_dst[new_key] = edgenameemb_dataset[key]
        elif key.startswith("tail of "):
            new_key = key.replace("tail of ", f"tail of {dataset_name}-")
            new_key = new_key.replace(":", f":{dataset_name}-")
            edgenameemb_dst[new_key] = edgenameemb_dataset[key]
        else:
            assert key == "fewshot"
    # edgenameemb_dst.update(edgenameemb_dataset)  # directly update the dictionary

    # combine featnameemb.pt files
    featnameemb_dataset = torch.load(os.path.join(dataset_path, "featnameemb.pt"))
    try:
        featnameemb_dst = torch.load(os.path.join(destination_path, "featnameemb.pt"))
    except FileNotFoundError:
        featnameemb_dst = dict()
    # featnameemb_dst = update_keys(featnameemb_dst, featnameemb_dataset, dataset_name)
    featnameemb_dst.update(featnameemb_dataset)  # directly update the dictionary

    # combine tasknameemb.pt files
    tasknameemb_dataset = torch.load(os.path.join(dataset_path, "tasknameemb.pt"))
    try:
        tasknameemb_dst = torch.load(os.path.join(destination_path, "tasknameemb.pt"))
    except FileNotFoundError:
        tasknameemb_dst = dict()
    tasknameemb_dst = update_keys(tasknameemb_dst, tasknameemb_dataset, dataset_name)
    # tasknameemb_dst.update(tasknameemb_dataset)  # directly update the dictionary


    # combine metaadj.yaml files
    metaadj_dataset = yaml.safe_load(open(os.path.join(dataset_path, "metaadj.yaml"), "r"))
    try:
        metaadj_dst = yaml.safe_load(open(os.path.join(destination_path, "metaadj.yaml"), "r"))
    except FileNotFoundError:
        metaadj_dst = dict()
    metaadj_dst = update_keys(metaadj_dst, metaadj_dataset, dataset_name)
    # metaadj_dst.update(metaadj_dataset)  # directly update the dictionary

    # combine metanode.yaml files
    metanode_dataset = yaml.safe_load(open(os.path.join(dataset_path, "metanode.yaml"), "r"))
    try:
        metanode_dst = yaml.safe_load(open(os.path.join(destination_path, "metanode.yaml"), "r"))
    except FileNotFoundError:
        metanode_dst = dict()
    metanode_dst = update_keys(metanode_dst, metanode_dataset, dataset_name)
    # metanode_dst.update(metanode_dataset)  # directly update the dictionary 

    # combine metatask.yaml files
    metatask_dataset = yaml.safe_load(open(os.path.join(dataset_path, "metatask.yaml"), "r"))
    try:
        metatask_dst = yaml.safe_load(open(os.path.join(destination_path, "metatask.yaml"), "r"))
    except FileNotFoundError:
        metatask_dst = dict()
    metatask_dst = update_keys(metatask_dst, metatask_dataset, dataset_name)
    # metatask_dst.update(metatask_dataset)  # directly update the dictionary

    # save the updated files
    torch.save(edgenameemb_dst, os.path.join(destination_path, "edgenameemb.pt"))
    torch.save(featnameemb_dst, os.path.join(destination_path, "featnameemb.pt"))
    torch.save(tasknameemb_dst, os.path.join(destination_path, "tasknameemb.pt"))
    yaml.safe_dump(metaadj_dst, open(os.path.join(destination_path, "metaadj.yaml"), "w"))
    yaml.safe_dump(metanode_dst, open(os.path.join(destination_path, "metanode.yaml"), "w"))
    yaml.safe_dump(metatask_dst, open(os.path.join(destination_path, "metatask.yaml"), "w"))

    # copy the edge directory
    edge_src = os.path.join(dataset_path, "edge")
    edge_dst = os.path.join(destination_path, "edge")
    # copy_subfolders(edge_src, edge_dst, dataset_name)
    for subfolder in os.listdir(edge_src):
        adj_src = os.path.join(edge_src, subfolder, "adj")
        adj_path = os.path.join(edge_dst, f"{dataset_name}-{subfolder}", "adj")
        dataset = hds.load_from_disk(adj_src).with_format("torch")
        # Rename features
        features = list(dataset.features.keys())
        rename_map = {}
        for key in features:
            if key.startswith("head of "):
                new_key = key.replace("head of ", f"head of {dataset_name}-")
                new_key = new_key.replace(":", f":{dataset_name}-")
                rename_map[key] = new_key
            elif key.startswith("tail of"):
                new_key = key.replace("tail of ", f"tail of {dataset_name}-")
                new_key = new_key.replace(":", f":{dataset_name}-")
                rename_map[key] = new_key
        if rename_map:
            dataset = dataset.rename_columns(rename_map)
            dataset.save_to_disk(adj_path)

    # copy the node directory
    node_src = os.path.join(dataset_path, "node")
    node_dst = os.path.join(destination_path, "node")
    copy_subfolders(node_src, node_dst, dataset_name)

    # copy the task directory
    task_src = os.path.join(dataset_path, "task")
    task_dst = os.path.join(destination_path, "task")
    copy_subfolders(task_src, task_dst, dataset_name)