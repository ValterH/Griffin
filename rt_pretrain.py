import torch
import torch.nn.functional as F
from hdataset import Graph, Task
from hloaderwrapper import LoaderWrapperTask
from hmodel import GriffinMod
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from hFloatEmb import SimpleRepeater, getfloatdec
import numpy as np
import accelerate
import argparse
import os.path as osp
from metric import compute_metric
import yaml

from roach.store import store

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
        for data in tqdm(loader, desc="Evaluating", disable=not accelerator.is_main_process):
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
        loss = F.mse_loss(output.flatten(), label.flatten().float())
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    graph = Graph(args.dataset, args.hiddim)
    task = Task(args.dataset, args.hiddim)

    tasknames = args.tasks
    eval_tasks = args.eval_tasks
    if len(tasknames) == 1:
        assert tasknames[0] in ["rel-hm-heldout", "rel-amazon-heldout", "rel-stack-heldout"]
        dataset_name = tasknames[0].split("-heldout")[0]
        with open("task_names.yaml", "r") as f:
            tasks_dict = yaml.load(f, Loader=yaml.FullLoader)
        tasknames = tasks_dict[tasknames[0]]

    if accelerator.is_main_process:
        print(tasknames)
        store_path = osp.expanduser(f"~/scratch/roach/stores/{args.date}")
        store.init(store_path)
        store.save({
            "script_name": "griffin",
            "seed": args.seed,
            "dataset": dataset_name,
            "model_dim": args.hiddim,
        }, "args")

    floatembmodel = SimpleRepeater(args.hiddim)

    dataset = construct_dataset(graph, task, tasknames, "train", args, floatembmodel)
    valid_dataset_dict = {
        taskname: construct_dataset(graph, task, [taskname], "valid", args, floatembmodel)
        for taskname in eval_tasks
    }
    test_dataset_dict = {
        taskname: construct_dataset(graph, task, [taskname], "test", args, floatembmodel)
        for taskname in eval_tasks
    }
    metric_dict = {
        taskname: task.metatask[taskname]["metric"] for taskname in eval_tasks
    }
    best_valid_metrics = {taskname: -1e9 for taskname in eval_tasks}
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
        model.train()
        if accelerator.is_main_process:
            print(f"Epoch {epoch} starts")
        dataset.rebuild_indice(accelerator)
        loader = DataLoader(
            dataset,
            shuffle=True,
            batch_size=1,
            collate_fn=lambda xlist: xlist[0],
            num_workers=16,
            prefetch_factor=4,
            persistent_workers=False,
            pin_memory=True
        )
        loader = accelerator.prepare(loader)
        for data in tqdm(loader, desc=f"Epoch {epoch}", disable=not accelerator.is_main_process):
            step += 1
            optimizer.zero_grad()
            loss = compute_loss(model, dec, data)
            accelerator.backward(loss)
            optimizer.step()

            if accelerator.is_main_process:
                store.log("step", step)
                store.log("epochs", step / len(loader))
            if step % 100 == 0:
                tbtracker.log({"training_loss": loss}, step=step)

            if step % args.eval_freq == 0:
                eval_metric = {}
                for taskname in eval_tasks:
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
                        print("DATASET", dataset_name)
                        task_name = taskname.replace(f"{dataset_name}-", "")
                        print(taskname, task_name)
                        tbtracker.log({f"valid_metric/{task_name}/{metric_dict[taskname]}": eval_metric[taskname]}, step=step)
                        print(f"steps: {step} valid_metric/{task_name}/{metric_dict[taskname]}: {eval_metric[taskname]}", flush=True)
                        split = "val"
                        k = f"{metric_dict[taskname]}/{dataset_name}/{task_name}/{split}"
                        store.log(k, eval_metric[taskname])
                        if eval_metric[taskname] > best_valid_metrics[taskname]:
                            best_valid_metrics[taskname] = eval_metric[taskname]
                            checkpoint_path = osp.join(args.savepath, f"checkpoint-{taskname}-best") if args.savepath is not None else None
                            if checkpoint_path is not None:
                                accelerator.save_model(model, checkpoint_path)
                                checkpoint_path = osp.join(args.savepath, f"checkpoint-{epoch}-{step}") if args.savepath is not None else None
                                accelerator.save_model(model, checkpoint_path)

            if step == args.max_steps:
                if accelerator.is_main_process:
                    print(f"Reached maximum steps {args.max_steps}, stopping training.")
                stopped = True
                break
        accelerator.wait_for_everyone()

        if stopped:
            break

    # test_metric = {}
    # if best_checkpoint_path is not None:
    #     if accelerator.is_main_process:
    #         print(f"Loading best checkpoint from {best_checkpoint_path}")
    #     unwrap_model = accelerator.unwrap_model(model)
    #     accelerate.load_checkpoint_in_model(unwrap_model, best_checkpoint_path)
    #     model = accelerator.prepare(unwrap_model)
    #     # resave the best checkpoint
    #     if accelerator.is_main_process:
    #         print(f"Saving best checkpoint at {osp.join(args.savepath, 'best_checkpoint')}")
    #         accelerator.save_model(model, osp.join(args.savepath, 'best_checkpoint'))
    # for taskname in tasknames:
    #     if accelerator.is_main_process:
    #         print(f"Testing {taskname}...")
    #     eval_metric = eval_task(
    #         model,
    #         dec,
    #         test_dataset_dict[taskname],
    #         args,
    #         accelerator,
    #         metric_dict[taskname],
    #     )
    #     if accelerator.is_main_process:
    #         tbtracker.log({f"test_metric/{taskname}": eval_metric}, step=step)
    #         print(f"test_metric/{taskname}: {eval_metric}")
    #         test_metric[taskname] = eval_metric

    # if accelerator.is_main_process:
    #     avg_metric = sum(test_metric.values()) / len(test_metric)
    #     print(f"Average test metric: {avg_metric}")

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
    parser.add_argument("--eval_tasks", type=str, nargs="+", default=["rel-amazon-item-churn"])
    parser.add_argument("--savepath", type=str, default=None)
    parser.add_argument("--loadpath", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--batchsize", type=int, default=512)
    parser.add_argument("--eval_batchsize", type=int)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=4e-4)
    parser.add_argument("--maxepoch", type=int, default=10_000)
    # parser.add_argument("--patience", type=int, default=-1)
    # parser.add_argument("--eval_per_epoch", type=int, default=3)

    parser.add_argument("--num_mp", type=int, default=4)
    parser.add_argument("--hiddim", type=int, default=256)
    parser.add_argument("--fanout", type=int, default=10)
    parser.add_argument("--fewshotfanout", type=int, default=3)
    parser.add_argument("--hop", type=int, default=2)
    parser.add_argument("--use_rev", type=str2bool, default=True)
    parser.add_argument("--use_gate", type=str2bool, default=True)

    #RT params
    parser.add_argument("--date", type=str, default="debug")
    parser.add_argument("--max_steps", type=int, default=50_000)
    parser.add_argument("--eval_freq", type=int, default=5000)

    args = parser.parse_args()
    args.eval_batchsize = args.batchsize if args.eval_batchsize is None else args.eval_batchsize
    # seed everything
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    main(args)