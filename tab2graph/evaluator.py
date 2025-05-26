import torch
import torch.nn.functional as F
import torchmetrics.functional as MF
import torchmetrics.retrieval as MR
import numpy as np
from dbinfer_bench import DBBTaskType, DBBTaskMeta

negated = lambda f: lambda *args, **kwargs: -f(*args, **kwargs)


def root_mean_squared_error(logits, target):
    return MF.mean_squared_error(logits, target, squared=False)


METRIC_FN = {
    "classification": {
        "accuracy": MF.accuracy,
        "ap": MF.average_precision,
        "auroc": MF.auroc,
        "f1": MF.f1_score,
        "hinge": negated(MF.hinge_loss),
        "recall": MF.recall,
    },
    "regression": {
        "mae": negated(MF.mean_absolute_error),
        "mse": negated(MF.mean_squared_error),
        "msle": negated(MF.mean_squared_log_error),
        "pearson": MF.pearson_corrcoef,
        "rmse": negated(root_mean_squared_error),
        "r2": MF.r2_score,
    },
    "retrieval": {
        "retrieval_auroc": "retrieval_auroc",
        "retrieval_mae": "retrieval_mae",
        "retrieval_logloss": "retrieval_logloss",
        "hr": MR.RetrievalHitRate(),
        "hr@1": MR.RetrievalHitRate(top_k=1),
        "mrr": MR.RetrievalMRR(),
        "ndcg": MR.RetrievalNormalizedDCG(),
    },
}

def get_metric_fn(meta : DBBTaskMeta):
    fn = METRIC_FN[meta.task_type][meta.evaluation_metric]
    if meta.task_type == DBBTaskType.classification:
        def _classification_wrapper(seeds, logits, labels, *args, **kwargs):
            # Shape:
            #   - logits: (N, C) or (N,) if C == 1
            #   - labels : (N,)
            with torch.no_grad():
                preds = F.softmax(logits, dim=1)
                if hasattr(meta, 'num_classes'):
                    num_classes = meta.num_classes
                else:
                    num_classes = preds.shape[1]
                return fn(preds, labels,
                          num_classes=num_classes,
                          task='multiclass')
        return _classification_wrapper
    elif meta.task_type == DBBTaskType.regression:
        def _regression_wrapper(seeds, logits, targets, *args, **kwargs):
            # Shape:
            #   - logits: (N,)
            #   - targets : (N,)
            with torch.no_grad():
                return fn(logits, targets.float())
        return _regression_wrapper
    elif meta.task_type == DBBTaskType.retrieval:
        def _retrieval_wrapper(query_idx, logits, labels, retrieve_target_labels=None, eval_trials=None, add_sigmoid=True, *args, **kwargs):
            # Shape:
            #   - query_idx: (N,)
            #   - logits: (N,)
            #   - labels : (N,)
            #   - retrieve_target_labels : (N / 2, ) or (N, ) if eval_trials is not None (i.e, valid/test)
            with torch.no_grad():
                if meta.evaluation_metric == 'retrieval_auroc':
                    # First check if all values in query_idx appear twice.
                    # If not, remove those that appear only once.
                    unique_query_idx, unique_counts = torch.unique(query_idx, return_counts=True)
                    # Create mask for elements appearing only once
                    remove_mask = torch.isin(query_idx, unique_query_idx[unique_counts == 1])
                    if torch.any(remove_mask):
                        remove_idx = torch.where(remove_mask)[0]
                        query_idx = query_idx[~remove_idx]
                        logits = logits[~remove_idx]
                        if retrieve_target_labels is not None:
                            retrieve_target_labels = retrieve_target_labels[~remove_idx]
                    # sort by query_idx
                    sorted_idx = query_idx.argsort(stable=True)
                    # remove the sorted_idx if unique value count
                    if add_sigmoid:
                        logits = torch.sigmoid(logits)
                    logits = logits[sorted_idx]
                    logits = torch.reshape(logits, (-1, 2))
                    if retrieve_target_labels is not None:
                        if retrieve_target_labels.shape[0] == query_idx.shape[0] // 2:
                            pass
                        else:
                            assert retrieve_target_labels.shape[0] == query_idx.shape[0]
                            # * Computing solution assumes that positive examples and negative examples are organized separately together.
                            # * This is a common pattern in retrieval tasks valid/test: In each trial,
                            # * the first half of the examples are positive examples, and the second half are negative examples. 
                            # Split the retrieve_target_labels into eval_trails parts and keep the first part
                            retrieve_target_labels = retrieve_target_labels[:retrieve_target_labels.shape[0] // 2]
                    # logits = F.softmax(logits, dim=1)
                    # Change the position of logits if corresponding value of retrieve_target is 1
                    logits_result = logits.clone().detach()
                    logits_result[retrieve_target_labels == 1] = logits[retrieve_target_labels == 1].flip(dims=(1,))
                    return MF.auroc(logits_result, retrieve_target_labels, num_classes=2, task='multiclass')
                elif meta.evaluation_metric == "retrieval_logloss":
                    # First check if all values in query_idx appear with number of classes.
                    # If not, remove those that appear less than number of classes.
                    unique_query_idx, unique_counts = torch.unique(query_idx, return_counts=True)
                    num_classes = max(unique_counts)
                    # Create mask for elements appearing less than number of classes
                    remove_mask = torch.isin(query_idx, unique_query_idx[unique_counts < num_classes])
                    if torch.any(remove_mask):
                        remove_idx = torch.where(remove_mask)[0]
                        query_idx = query_idx[~remove_idx]
                        logits = logits[~remove_idx]
                        if retrieve_target_labels is not None:
                            retrieve_target_labels = retrieve_target_labels[~remove_idx]
                    # Sort the logits and query_idx for consistency
                    sorted_idx = query_idx.argsort(stable=True)
                    num_samples = len(np.unique(query_idx))
                    classes = int(len(query_idx) / num_samples)
                    assert classes == num_classes
                    # Apply sigmoid to logits to convert them to probabilities
                    if add_sigmoid:
                        logits = torch.sigmoid(logits)
                    logits = logits[sorted_idx]
                    labels = labels[sorted_idx]
                    labels = torch.reshape(labels, (-1, classes))
                    # Reshape logits to match the number of examples
                    logits = torch.reshape(logits, (-1, classes))
                    # Prepare for log loss calculation
                    # Log loss expects probability distribution over classes and true labels
                    log_probs = torch.log(logits + 1e-7)  # Adding small epsilon to avoid log(0)
                    # Compute the negative log likelihood loss (log loss)
                    # Cross-entropy between predicted probabilities and true labels
                    log_loss = -torch.sum(labels * log_probs) / num_samples

                    return -log_loss

                preds = torch.sigmoid(logits)
                return fn(preds, labels, indexes=query_idx)
        return _retrieval_wrapper
    else:
        raise ValueError(f"Unsupported task type {meta.task_type}")

def infer_task_type(metric_name: str):
    for task_type, metrics in METRIC_FN.items():
        if metric_name in metrics:
            return task_type
    raise ValueError(f"Invalid metric name {metric_name}.")

def retrieval_loss(logits, labels):
    # num_pos = (labels == 1).sum()
    # num_neg = (labels == 0).sum()
    # weight = torch.where(labels == 1, num_neg / (num_pos + num_neg), num_pos / (num_pos + num_neg))
    return F.binary_cross_entropy(torch.sigmoid(logits), labels.float())

LOSS_FN = {
    'classification': F.cross_entropy,
    'regression': lambda logits, targets : F.mse_loss(logits, targets.float()),
    'retrieval': retrieval_loss,
}

def get_loss_fn(meta : DBBTaskMeta):
    # if meta.evaluation_metric == "retrieval_auroc":
    #     def _retrieval_auroc_loss(logits, labels):
    #         logits = logits.view(-1, 2)
    #         # labels = labels.view(-1, 2)
    #         pos_idx = torch.ones(logits.shape[0], dtype=torch.long)
    #         # pos_idx = labels.argmax(dim=1)
    #         return F.cross_entropy(logits, pos_idx)
    fn = LOSS_FN[meta.task_type]
    return fn
