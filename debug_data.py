import argparse
import os.path as osp

import yaml
import accelerate
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F

from hdataset import Graph, Task
from hloaderwrapper import LoaderWrapperTask
from hmodel import GriffinMod
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from hFloatEmb import SimpleRepeater, getfloatdec
import metric
from roach.store import store

from metric import compute_metric

def eval_task(model, dec, dataset, args, accelerator, metric):
    model.eval()
    dataset.rebuild_indice(accelerator)
    batchsize: int = dataset.batch_size
    loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        collate_fn=lambda xlist: xlist[0],
        num_workers=8,
        persistent_workers=False,
    )
    loader = accelerator.prepare(loader)
    outputs = []
    labels = []
    with torch.no_grad():
        for data in tqdm(loader, desc="Val", disable=not accelerator.is_main_process):
            output, label = compute_output(model, dec, data)
            if output.shape[0] < batchsize:
                assert output.ndim == 2
                assert label.ndim == 1
                padnum = batchsize-output.shape[0]
                if torch.is_floating_point(label):
                    label = torch.concat((label, torch.empty_like(label[[0]].expand(padnum)).fill_(torch.nan)), dim=0)
                else:
                    label = torch.concat((label, torch.empty_like(label[[0]].expand(padnum)).fill_(-1)), dim=0)
                output = torch.concat((output, torch.empty_like(output[[0]].expand(padnum, -1)).fill_(torch.nan)), dim=0)
            output, label = output.unsqueeze(0), label.unsqueeze(0)
            output, label = accelerator.gather_for_metrics((output, label))
            output, label = output.flatten(0, 1), label.flatten(0, 1)
            if accelerator.is_main_process:
                if torch.is_floating_point(label):
                    mask = torch.isnan(label).logical_not_()
                else:
                    mask = label >= 0
                output, label = output[mask], label[mask]
                outputs.append(output.cpu())
                labels.append(label.cpu())
    #if metric.requires_gather:
    #    metric.outputs, metric.labels = accelerator.gather_for_metrics((metric.outputs, metric.labels))
    if accelerator.is_main_process:
        labels = torch.concat(labels, dim=0)
        outputs = torch.concat(outputs, dim=0)
        return compute_metric(outputs, labels, metric)
    else:
        return None


def construct_dataset(graph, task, tasknames, split, args, floatembmodel):
    return LoaderWrapperTask(
        graph,
        batch_size=args.batchsize,
        subgraphargs={
            "floatemb": floatembmodel,
            "fanout": args.fanout,
            "hop": args.hop,
        },
        shuffle=True if split == "train" else False,
        task=task,
        tasknames=tasknames,
        split=split,
        fewshotfanout=args.fewshotfanout,
    )


def compute_output(model, dec, data):
    label, y, mapping = data[-3:]
    data = data[:-3]
    if y is None:
        output = dec(model(*data)[mapping])
    else:
        output = model(*data)[mapping] @ y.T
    return output, label


def compute_loss(model, dec, data):
    label, y, mapping = data[-3:]
    data = data[:-3]
    if y is None:
        output = dec(model(*data)[mapping])
        loss = F.mse_loss(output.flatten(), label.flatten())
    else:
        output = model(*data)[mapping] @ y.T
        loss = F.cross_entropy(output, label)
    return loss


def main(args):
    tbconfig = ProjectConfiguration(project_dir=args.logdir, logging_dir=args.logdir)
    accelerator = Accelerator(log_with="tensorboard", project_config=tbconfig)
    accelerator.init_trackers(args.logname)
    tbtracker = accelerator.get_tracker("tensorboard")

    model = GriffinMod(hiddim=args.hiddim, num_mp=args.num_mp, use_rev=args.use_rev, use_gate=args.use_gate)
    if args.loadpath is not None:
        accelerate.load_checkpoint_in_model(model, args.loadpath)
    # model.reset_parameters()
    dec = getfloatdec(args.hiddim)

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if accelerator.is_main_process:
        print(f"Number of trainable parameters: {num_parameters}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    graph = Graph(args.dataset)
    task = Task(args.dataset)

    tasknames = args.tasks
    if len(tasknames) == 1:
        if tasknames[0] == "ALLTASK":
            tasknames = [taskname for taskname in task.metatask]
        elif tasknames[0] == "RETTASK":
            tasknames = [
                taskname
                for taskname in task.metatask
                if task.metatask[taskname]["task_type"] == "retrieval"
            ]
        elif tasknames[0] == "REGTASK":
            tasknames = [
                taskname
                for taskname in task.metatask
                if task.metatask[taskname]["task_type"] == "regression"
            ]
        elif tasknames[0].startswith("EXCEPT__"):
            expect_taskname = tasknames[0][len("EXCEPT__"):]
            tasknames = [taskname for taskname in task.metatask if taskname != expect_taskname]
        elif tasknames[0] in ["commerce-1", "commerce-2", "others-1", "others-2"]:
            with open("task_names.yaml", "r") as f:
                tasks_dict = yaml.load(f, Loader=yaml.FullLoader)
            tasknames = tasks_dict[tasknames[0]]

    if accelerator.is_main_process:
        print(tasknames)
        assert len(tasknames) == 1, f"Expected one task name, but got {len(tasknames)}"
        store_path = osp.expanduser(f"~/scratch/roach/stores/{args.date}")
        store.init(store_path)
        # e.g. taskname = "rel-hm-user-churn"
        dashed = tasknames[0].split("-")
        dataset_name = f"{dashed[0]}-{dashed[1]}" # rel-hm
        task_name = "-".join(dashed[2:]) # user-churn
        pretrained = None
        if args.loadpath is not None:
            if "single-sft" in args.loadpath:
                pretrained = "single-sft"
            else:
                raise ValueError(f"Unknown loadpath {args.loadpath}")
        store.save({
            "script_name": "griffin",
            "seed": args.seed,
            "dataset": dataset_name,
            "task": task_name,
            "pretrain_steps": pretrained
        }, "args")

    floatembmodel = SimpleRepeater(args.hiddim)

    dataset = construct_dataset(graph, task, tasknames, "train", args, floatembmodel)
    valid_dataset_dict = {
        taskname: construct_dataset(graph, task, [taskname], "valid", args, floatembmodel)
        for taskname in tasknames
    }
    test_dataset_dict = {
        taskname: construct_dataset(graph, task, [taskname], "test", args, floatembmodel)
        for taskname in tasknames
    }
    metric_dict = {
        taskname: task.metatask[taskname]["metric"] for taskname in tasknames
    }
    best_valid_metric = -torch.inf
    best_checkpoint_path = None
    best_epoch = 0

    model, dec, optimizer = accelerator.prepare(model, dec, optimizer)

    if args.mode == "test":
        test_metric = {}
        for taskname in tasknames:
            if accelerator.is_main_process:
                print(f"test {taskname}...")
            eval_metric = eval_task(
                model,
                dec,
                test_dataset_dict[taskname],
                args,
                accelerator,
                metric_dict[taskname],
            )
            test_metric[taskname] = eval_metric
            if accelerator.is_main_process:
                print(f"test_metric/{taskname}: {test_metric[taskname]}", flush=True)
        accelerator.end_training()
        return

    model.train()
    step = 0
    stopped = False
    for epoch in range(args.maxepoch):
        if accelerator.is_main_process:
            print(f"Epoch {epoch} starts")
        dataset.rebuild_indice(accelerator)
        loader = DataLoader(
            dataset,
            shuffle=True,
            batch_size=1,
            collate_fn=lambda xlist: xlist[0],
            num_workers=0,
            prefetch_factor=None,
            persistent_workers=False,
            pin_memory=True
        )
        loader = accelerator.prepare(loader)
        for data in tqdm(loader, desc="Train", disable=not accelerator.is_main_process):
            if step & (step - 1) == 0:
                eval_metric = {}
                for taskname in tasknames:
                    if accelerator.is_main_process:
                        print(f"Validating {taskname}...")
                    eval_metric[taskname] = eval_task(
                        model,
                        dec,
                        valid_dataset_dict[taskname],
                        args,
                        accelerator,
                        metric_dict[taskname],
                    )

                    if accelerator.is_main_process:
                        store.log("step", step)
                        store.log("epochs", step / len(loader))
                        tbtracker.log({f"valid_metric/{taskname}/{metric_dict[taskname]}": eval_metric[taskname]}, step=step)
                        print(f"steps: {step} valid_metric/{taskname}/{metric_dict[taskname]}: {eval_metric[taskname]}", flush=True)
                        split = "val"
                        k = f"{metric_dict[taskname]}/{dataset_name}/{task_name}/{split}"
                        store.log(k, eval_metric[taskname])
                model.train()
            step += 1
            if step == args.max_steps:
                if accelerator.is_main_process:
                    print(f"Reached maximum steps {args.max_steps}, stopping training.")
                stopped = True
                break
            optimizer.zero_grad()
            loss = compute_loss(model, dec, data)
            accelerator.backward(loss)
            optimizer.step()
            if step % 100 == 0:
                tbtracker.log({"training_loss": loss}, step=step)
            
        if stopped:
            if accelerator.is_main_process:
                print(f"Training stopped at step {step} due to reaching max steps.")
            break
        accelerator.wait_for_everyone()
        checkpoint_path = osp.join(args.savepath, f"checkpoint-{epoch}-{step}") if args.savepath is not None else None
        if args.savepath is not None:
            if accelerator.is_main_process:
                accelerator.save_model(model, checkpoint_path)


    accelerator.end_training()

if __name__ == "__main__":
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", help="train or test")
    parser.add_argument("dataset", type=str)
    parser.add_argument("logdir", type=str)
    parser.add_argument("logname", type=str)
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["ALLTASK"],
        help="ALLTASK for all tasks. RETTASK for all retrieval task. REGTASK for all regression task. Otherwise input a list of task name",
    )
    parser.add_argument("--savepath", type=str, default=None)
    parser.add_argument("--loadpath", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--batchsize", type=int, default=512)
    parser.add_argument("--eval_batchsize", type=int)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=4e-4)
    parser.add_argument("--maxepoch", type=int, default=100)
    parser.add_argument("--patience", type=int, default=-1)
    parser.add_argument("--eval_per_epoch", type=int, default=3)

    parser.add_argument("--num_mp", type=int, default=4)
    parser.add_argument("--hiddim", type=int, default=256)
    parser.add_argument("--fanout", type=int, default=10)
    parser.add_argument("--fewshotfanout", type=int, default=3)
    parser.add_argument("--hop", type=int, default=2)
    parser.add_argument("--use_rev", type=str2bool, default=True)
    parser.add_argument("--use_gate", type=str2bool, default=True)
    # relfm reproducibility args
    parser.add_argument("--date", type=str, default="2025-07-30", help="Date for the store path")
    parser.add_argument("--max_steps", type=int, default=8192, help="Maximum number of steps to run")

    args = parser.parse_args()
    args.eval_batchsize = args.batchsize if args.eval_batchsize is None else args.eval_batchsize
    # seed everything
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    main(args)
