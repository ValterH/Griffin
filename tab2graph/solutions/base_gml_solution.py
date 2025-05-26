import abc
import copy
import logging
from typing import Tuple, Dict, Optional, List, Any
from pathlib import Path
import pydantic
import tqdm
import time
import os
import wandb
import functools
from collections import defaultdict
import random

import dgl
import dgl.graphbolt as gb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.distributed.algorithms.join import Join
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

import numpy as np
from sklearn.neighbors import KernelDensity

from dbinfer_bench import (
    DBBGraphDataset,
    DBBTaskType,
    DBBTaskMeta,
    DBBGraphTaskMeta,
    TIMESTAMP_FEATURE_NAME,
)

from .base import (
    GraphMLSolution,
    GraphMLSolutionConfig,
    FitSummary,
)
from .graph_dataset_config import (
    GraphConfig,
    GraphDatasetConfig,
    GraphDatasetMultiTaskConfig,
)
from .encoders import GraphFeatDictEncoder, GraphFeatDictMultiTaskEncoder, IdDictEncoder, SelfAttentionAggregator
from .predictor import PredictorConfig, Predictor, SeedLookup
from .negative_sampler import DBBNegativeSampler
from ..evaluator import get_metric_fn, get_loss_fn
from ..device import DeviceInfo
from .. import yaml_utils
from ..time_budget import TimeBudgetedIterator
from ..sample_utils import StrictType

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

NType = str
EType = Tuple[str, str, str]

__all__ = ['BaseGMLSolution']


class FewshotSampler(pydantic.BaseModel):
    method = 'KDE'
    samples_list: List = [0, 8, 16, 32, 64, 128, 256, 512]
    bandwidth_list: List = [0.5, 0.125, 0.0625, 0.03125, 0.001, 0.0001, 0.00001, 0.000001]


fewshot_sampler = FewshotSampler()

class SHUFFLE_STRATEGY:
    # shuffle_strategy includes three levels:
    # 1. DEFAULT: no shuffling
    # 2. RANDOM_TASK: shuffle tasks, but will not shuffle samples across each task,
    # which means we will finish one task before moving to the next one.
    # 3. RANDOM_SAMPLE: shuffle both tasks and samples,
    # which means we will shuffle samples across tasks.
    DEFAULT = 'default'
    RANDOM_TASK = 'random_task'
    RANDOM_SAMPLE = 'random_sample'


class DynamicSampleStrategy(pydantic.BaseModel):
    epoch_num_per_update : int = 3
    update_ratio : float = 0.2
    update_percentage : float = 0.3
    clip_ratio_lowerbound : float = 0.3
    clip_ratio_upperbound : float = 2.0
    target_metrics_dict : Dict[str, float] = None


class BaseGNNSolutionConfig(GraphMLSolutionConfig):
    predictor: Optional[PredictorConfig] = PredictorConfig()
    use_multiprocessing: bool = True
    eval_trials: int = 10
    strict_mode: Optional[StrictType] = StrictType.full
    # with_original_features means whether the data configs including raw numerical or categorical features.
    # These features should not be encoded by the encoder, but only used for the ID invariance checker.
    with_original_features: Optional[bool] = False
    llm_dim: Optional[int] = 256
    use_one_encoder_for_llm: Optional[bool] = False
    use_one_decoder_for_llm: Optional[bool] = False
    add_columns_dim_together: Optional[bool] = True
    use_attention_aggregation: Optional[bool] = False
    attention_num_layer: Optional[int] = 1
    attention_num_heads: Optional[int] = 4
    batch_size_for_each_task: Optional[Dict[str, int]] = {}
    sample_ratio_for_each_task: Optional[Dict[str, float]] = {}
    shuffle_strategy: Optional[str] = "default"
    use_lr_scheduler: Optional[bool] = False
    lr_scheduler_ratio: Optional[float] = 0.5
    lr_scheduler_patience: Optional[int] = 3
    # For params in dynamic_sample_strategy, see parse_dynamic_sample_strategy
    dynamic_sample_strategy: Optional[DynamicSampleStrategy] = None
    not_allow_leakage_features: Optional[bool] = True

def parse_dynamic_sample_strategy(dynamic_sample_strategy):
    epoch_num_per_update = dynamic_sample_strategy.epoch_num_per_update
    update_ratio = dynamic_sample_strategy.update_ratio
    update_percentage = dynamic_sample_strategy.update_percentage
    clip_ratio_lowerbound = dynamic_sample_strategy.clip_ratio_lowerbound
    clip_ratio_upperbound = dynamic_sample_strategy.clip_ratio_upperbound
    target_metrics_dict = dynamic_sample_strategy.target_metrics_dict
    if target_metrics_dict is None:
        AssertionError("target_metrics_dict must be specified for dynamic sample strategy.")
    return epoch_num_per_update, update_ratio, update_percentage, clip_ratio_lowerbound, clip_ratio_upperbound, target_metrics_dict

class BaseGNN(nn.Module):

    def __init__(
        self,
        solution_config : BaseGNNSolutionConfig,
        data_config : GraphDatasetConfig
    ):
        super().__init__()
        self._solution_config = solution_config
        self._data_config = data_config

        self.feat_encoder = GraphFeatDictEncoder(
            data_config,
            solution_config.feat_encode_size)
        self.node_id_encoder = IdDictEncoder(
            data_config.graph.num_nodes,
            solution_config.embed_ntypes,
            solution_config.feat_encode_size
        )
        node_out_size_dict = dict(self.node_id_encoder.out_size_dict)
        node_out_size_dict.update(self.feat_encoder.node_out_size_dict)
        for ntype in solution_config.embed_ntypes:
            if ntype in self.feat_encoder.node_out_size_dict:
                node_out_size_dict[ntype] += self.node_id_encoder.out_size_dict[ntype]

        if solution_config.predictor is None:
            assert data_config.task.num_seeds == 1, \
                "Setting predictor to be None is only allowed for node-level prediction."
            gnn_out_size = Predictor.get_out_size(data_config.task)
        else:
            gnn_out_size = None
        self.gnn = self.create_gnn(
            node_out_size_dict,
            self.feat_encoder.edge_out_size_dict,
            self.feat_encoder.seed_ctx_out_size,
            gnn_out_size)

        self.seed_lookup = SeedLookup(data_config.task.seed_type)
        if solution_config.predictor is None:
            self.predictor = lambda seed_embeds, seed_ctx_embeds : seed_embeds
        else:
            self.predictor = Predictor(
                data_config.task,
                solution_config.predictor,
                self.gnn.out_size,
                self.feat_encoder.seed_ctx_out_size,
            )

    @property
    def solution_config(self) -> BaseGNNSolutionConfig:
        return self._solution_config

    @property
    def data_config(self) -> GraphDatasetConfig:
        return self._data_config

    @abc.abstractmethod
    def create_gnn(
        self,
        graph_cfg : GraphConfig,
        node_feat_size_dict : Dict[str, int],
        edge_feat_size_dict : Dict[str, int],
        seed_feat_size : int,
        out_size : Optional[int],
    ) -> nn.Module:
        pass

    def forward(
        self,
        mfgs,
        node_feat_dict : Dict[str, Dict[str, torch.Tensor]],
        input_node_id_dict : Dict[str, torch.Tensor],
        edge_feat_dicts : List[Dict[str, Dict[str, torch.Tensor]]],
        seed_feat_dict : Dict[str, Dict[str, torch.Tensor]],
        seed_lookup_idx : torch.Tensor
    ):
        # Encode IDs.
        H_id_dict = self.node_id_encoder(input_node_id_dict) # for each id type, int id -> emb

        # Encode features.
        H_feat_dict = self.feat_encoder(node_feat_dict) # float to float, int to emb, according to type

        # Mask leakage features. They are features of target type that exists
        # in RDB but not in seed contexts.
        target_type = self.data_config.task.target_type
        seed_type = self.data_config.task.seed_type

        # ?? move mask to data preprocess
        # use attention mask rather than set 0
        if seed_type == target_type and target_type in H_feat_dict:
            # Mask out features of seeds.
            neigh_feat_set = set(H_feat_dict[target_type].keys())
            seed_feat_set = set(seed_feat_dict['__seed__'].keys())
            num_seeds = mfgs[-1].num_dst_nodes(ntype=target_type)
            for key_to_mask in neigh_feat_set - seed_feat_set:
                H = H_feat_dict[target_type][key_to_mask]
                H[:num_seeds] = 0.

        H_feat_dict = _cat_feat(H_feat_dict)# id name, col name, value -> id name, value

        # Merge two dictionaries.
        H_node_dict = dict(H_id_dict)
        H_node_dict.update(H_feat_dict)
        for ntype in self.solution_config.embed_ntypes:
            if ntype in self.feat_encoder.node_out_size_dict:
                H_node_dict[ntype] = torch.cat(
                    [H_feat_dict[ntype], H_id_dict[ntype]], dim=1)

        # Encode edges.
        H_edge_dicts = [
            _cat_feat(self.feat_encoder(edge_feat_dict))
            for edge_feat_dict in edge_feat_dicts
        ]

        # Message passing.
        H_node_dict = self.gnn(mfgs, H_node_dict, H_edge_dicts)

        # Prediction head.
        # TODO: Check if order for training is fair!
        seed_embeds = self.seed_lookup(H_node_dict, seed_lookup_idx)
        seed_ctx_embeds = _cat_feat(self.feat_encoder(seed_feat_dict))['__seed__']
        return self.predictor(seed_embeds, seed_ctx_embeds)

    def get_node_embeddings(self) -> Dict[str, torch.Tensor]:
        return self.node_id_encoder.get_embedding_dict()


class BaseMultiTaskGNN(nn.Module):

    def __init__(
        self,
        solution_config : BaseGNNSolutionConfig,
        data_config : GraphDatasetMultiTaskConfig,
    ):
        super().__init__()
        self._solution_config = solution_config
        self._data_config = data_config

        self.feat_encoder = GraphFeatDictMultiTaskEncoder(
            self._data_config,
            solution_config.feat_encode_size,
            solution_config.use_one_encoder_for_llm,
            solution_config.add_columns_dim_together,
            solution_config.with_original_features,
            solution_config.llm_dim,
        )
        self.node_id_encoder = IdDictEncoder(
            self._data_config.graph.num_nodes,
            solution_config.embed_ntypes,
            solution_config.feat_encode_size
        )
        node_out_size_dict = dict(self.node_id_encoder.out_size_dict)
        node_out_size_dict.update(self.feat_encoder.node_out_size_dict)
        for ntype in solution_config.embed_ntypes:
            if ntype in self.feat_encoder.node_out_size_dict:
                node_out_size_dict[ntype] += self.node_id_encoder.out_size_dict[ntype]

        if solution_config.predictor is None:
            AssertionError("Predictor must be specified for multi-task learning.")
            assert data_config.task.num_seeds == 1, \
                "Setting predictor to be None is only allowed for node-level prediction."
            gnn_out_size = Predictor.get_out_size(data_config.task)
        else:
            gnn_out_size = None
        self.gnn = self.create_gnn(
            node_out_size_dict,
            self.feat_encoder.edge_out_size_dict,
            gnn_out_size)

        self.seed_lookup_dict = nn.ModuleDict()
        for task_name in data_config.task_dict.keys():
            self.seed_lookup_dict[task_name] = SeedLookup(data_config.task_dict[task_name].seed_type)

        if solution_config.predictor is None:
            AssertionError("Predictor must be specified for multi-task learning.")
            self.predictor = lambda seed_embeds, seed_ctx_embeds : seed_embeds
        else:
            self.predictor_dict = nn.ModuleDict()
            if solution_config.use_one_decoder_for_llm:
                # Generate a retriever predictor and a regressor predictor
                # find a retriever task and a regressor task
                retriever_task_name = None
                regressor_task_name = None
                for task_name in data_config.task_dict.keys():
                    if data_config.task_dict[task_name].task_type == DBBTaskType.retrieval:
                        retriever_task_name = task_name
                    elif data_config.task_dict[task_name].task_type == DBBTaskType.regression:
                        regressor_task_name = task_name

                if retriever_task_name is not None:
                    Shared_retriever_predictor = Predictor(
                        data_config.task_dict[retriever_task_name],
                        solution_config.predictor,
                        self.gnn.out_size,
                        self.feat_encoder.seed_ctx_out_size_dict[retriever_task_name],
                    )
                if regressor_task_name is not None:
                    Shared_regressor_predictor = Predictor(
                        data_config.task_dict[regressor_task_name],
                        solution_config.predictor,
                        self.gnn.out_size,
                        self.feat_encoder.seed_ctx_out_size_dict[regressor_task_name],
                    )
                for task_name in data_config.task_dict.keys():
                    if data_config.task_dict[task_name].task_type == DBBTaskType.retrieval:
                        self.predictor_dict[task_name] = Shared_retriever_predictor
                    elif data_config.task_dict[task_name].task_type == DBBTaskType.regression:
                        self.predictor_dict[task_name] = Shared_regressor_predictor
            else:
                for task_name in data_config.task_dict.keys():
                    self.predictor_dict[task_name] = Predictor(
                        data_config.task_dict[task_name],
                        solution_config.predictor,
                        self.gnn.out_size,
                        self.feat_encoder.seed_ctx_out_size_dict[task_name],
                    )

        if solution_config.use_attention_aggregation:
            self.attention_model = SelfAttentionAggregator(
                in_size=solution_config.feat_encode_size,
                out_size=solution_config.feat_encode_size,
                num_layer=solution_config.attention_num_layer,
                num_heads=solution_config.attention_num_heads,
                dim_feedforward=solution_config.feat_encode_size,  # solution_config.row_encode_size
            )

    @property
    def solution_config(self) -> BaseGNNSolutionConfig:
        return self._solution_config

    @property
    def data_config(self) -> GraphDatasetConfig:
        return self._data_config

    @abc.abstractmethod
    def create_gnn(
        self,
        graph_cfg : GraphConfig,
        node_feat_size_dict : Dict[str, int],
        edge_feat_size_dict : Dict[str, int],
        seed_feat_size : int,
        out_size : Optional[int],
    ) -> nn.Module:
        pass


    def forward(
        self,
        mfgs, # list of dgl block
        node_feat_dict : Dict[str, Dict[str, torch.Tensor]],  # id name, column name, int/float (256,*) or (0, 1) or (0, 256)
        input_node_id_dict : Dict[str, torch.Tensor],  # id name, int tensor (256) or empty
        edge_feat_dicts : List[Dict[str, Dict[str, torch.Tensor]]],  # empty list
        seed_feat_dict : Dict[str, Dict[str, torch.Tensor]],  # ?, column name, float tensor (256, *)
        seed_lookup_idx : torch.Tensor,  # int (256, 2)
        task_name: str,
        task: DBBGraphTaskMeta,
        model_device
    ):
        task_emb = torch.tensor(task.task_emb, device=model_device)
        # Encode IDs.
        H_id_dict = self.node_id_encoder(input_node_id_dict)

        # Encode features.
        # Before encoding, the node_feat_dict already only contains the features of Griffin_text
        H_feat_dict = self.feat_encoder(node_feat_dict)

        # Mask leakage features. They are features of target type that exists
        # in RDB but not in seed contexts.
        if self._solution_config.not_allow_leakage_features:
            target_type = task.target_type
            seed_type = task.seed_type
            if seed_type == target_type and target_type in H_feat_dict:
                # Mask out features of seeds.
                neigh_feat_set = set(H_feat_dict[target_type].keys())
                seed_feat_set = set(seed_feat_dict[f'__seed__-{task_name}'].keys())
                if mfgs:
                    num_seeds = mfgs[-1].num_dst_nodes(ntype=target_type)
                else:
                    num_seeds = self._solution_config.batch_size
                for key_to_mask in neigh_feat_set - seed_feat_set:
                    H = H_feat_dict[target_type][key_to_mask]
                    H[:num_seeds] = 0.
            # * A fix patch for retrieval task
            if task.task_type == DBBTaskType.retrieval:
                target_type_src = task.seed_type.split(':')[0]
                if target_type == target_type_src and target_type_src in H_feat_dict:
                    neigh_feat_set = set(H_feat_dict[target_type_src].keys())
                    seed_feat_set = set(seed_feat_dict[f'__seed__-{task_name}'].keys())
                    if mfgs:
                        num_seeds = mfgs[-1].num_dst_nodes(ntype=target_type_src)
                    else:
                        num_seeds = self._solution_config.batch_size
                    for key_to_mask in neigh_feat_set - seed_feat_set:
                        H = H_feat_dict[target_type_src][key_to_mask]
                        H[:num_seeds] = 0.

        if self._solution_config.use_attention_aggregation and self._solution_config.with_original_features:
            H_feat_dict, H_relational_feat_dict = _attention_merge_feat_with_relation(
                H_feat_dict, task_emb, self.data_config.node_features, self.attention_model, model_device=model_device
            )
        elif self._solution_config.use_attention_aggregation:
            H_feat_dict = _attention_merge_feat(
                H_feat_dict, task_emb, self.attention_model
            )
        elif self._solution_config.add_columns_dim_together:
            H_feat_dict = _cat_feat(H_feat_dict)
        else:
            H_feat_dict = _merge_feat(H_feat_dict)

        # Merge two dictionaries.
        H_node_dict = dict(H_id_dict)
        H_node_dict.update(H_feat_dict)
        for ntype in self.solution_config.embed_ntypes:
            if ntype in self.feat_encoder.node_out_size_dict:
                H_node_dict[ntype] = torch.cat(
                    [H_feat_dict[ntype], H_id_dict[ntype]], dim=1)

        # Encode edges.
        if self._solution_config.use_attention_aggregation:
            # * In practice, all edge features are set to none.
            # * This part is only for placeholder.
            H_edge_dicts = [
                _attention_merge_feat(
                    self.feat_encoder(edge_feat_dict),
                    task_emb,
                    self.attention_model,
                )
                for edge_feat_dict in edge_feat_dicts
            ]
        elif self._solution_config.add_columns_dim_together:
            H_edge_dicts = [
                _cat_feat(self.feat_encoder(edge_feat_dict))
                for edge_feat_dict in edge_feat_dicts
            ]
        else:
            H_edge_dicts = [
                _merge_feat(self.feat_encoder(edge_feat_dict))
                for edge_feat_dict in edge_feat_dicts
            ]

        # Message passing.
        if self._solution_config.with_original_features and self._solution_config.use_attention_aggregation:
            H_node_dict = self.gnn(mfgs, H_node_dict, H_edge_dicts, H_relational_feat_dict)
        else:
            H_node_dict = self.gnn(mfgs, H_node_dict, H_edge_dicts)

        # Prediction head.
        seed_embeds = self.seed_lookup_dict[task_name](H_node_dict, seed_lookup_idx)
        if self._solution_config.predictor.with_seed_ctx_features:
            if (
                self._solution_config.use_attention_aggregation
                and self._solution_config.with_original_features
            ):
                seed_ctx_embeds = _attention_merge_feat_with_relation(
                    self.feat_encoder(seed_feat_dict),
                    task_emb,
                    self.data_config.seed_features_dict,
                    self.attention_model,
                    return_relation_feat=False,
                    model_device=model_device
                )[f"__seed__-{task_name}"]
            elif self._solution_config.use_attention_aggregation:
                seed_ctx_embeds = _attention_merge_feat(
                    self.feat_encoder(seed_feat_dict), self.attention_model
                )[f"__seed__-{task_name}"]
            elif self._solution_config.add_columns_dim_together:
                seed_ctx_embeds = (
                    _cat_feat(self.feat_encoder(seed_feat_dict))[f"__seed__-{task_name}"]
                )
            else:
                seed_ctx_embeds = (
                    _merge_feat(self.feat_encoder(seed_feat_dict))[f"__seed__-{task_name}"]
                )
        else:
            seed_ctx_embeds = None
        del task_emb
        return self.predictor_dict[task_name](seed_embeds, seed_ctx_embeds)


    def get_seed_embeds(
        self,
        mfgs,
        node_feat_dict : Dict[str, Dict[str, torch.Tensor]],
        input_node_id_dict : Dict[str, torch.Tensor],
        edge_feat_dicts : List[Dict[str, Dict[str, torch.Tensor]]],
        seed_feat_dict : Dict[str, Dict[str, torch.Tensor]],
        seed_lookup_idx : torch.Tensor,
        task_name: str,
        task
    ):
        # prepare possible return embeddings
        llm_embeds, encoder_embeds = None, None

        # Encode IDs.
        H_id_dict = self.node_id_encoder(input_node_id_dict)

        # if we use one sentence representing the whole role, the embedding can be kept here
        if not self._solution_config.use_attention_aggregation and self._solution_config.add_columns_dim_together:
            llm_embeds = node_feat_dict[task.target_type]['Griffin_text']
            llm_embeds = llm_embeds[seed_lookup_idx[:, 0]]

        # Encode features.
        H_feat_dict = self.feat_encoder(node_feat_dict)

        # Mask leakage features. They are features of target type that exists
        # in RDB but not in seed contexts.
        target_type = task.target_type
        seed_type = task.seed_type
        if seed_type == target_type and target_type in H_feat_dict:
            # Mask out features of seeds.
            neigh_feat_set = set(H_feat_dict[target_type].keys())
            seed_feat_set = set(seed_feat_dict[f'__seed__-{task_name}'].keys())
            num_seeds = mfgs[-1].num_dst_nodes(ntype=target_type)
            for key_to_mask in neigh_feat_set - seed_feat_set:
                H = H_feat_dict[target_type][key_to_mask]
                H[:num_seeds] = 0.
        # * A fix patch for retrieval task
        if task.task_type == DBBTaskType.retrieval:
            #TODO: Should be seed_type instead of target_type
            target_type_src = task.seed_type.split(':')[0]
            if target_type == target_type_src and target_type_src in H_feat_dict:
                neigh_feat_set = set(H_feat_dict[target_type_src].keys())
                seed_feat_set = set(seed_feat_dict[f'__seed__-{task_name}'].keys())
                num_seeds = mfgs[-1].num_dst_nodes(ntype=target_type_src)
                for key_to_mask in neigh_feat_set - seed_feat_set:
                    H = H_feat_dict[target_type_src][key_to_mask]
                    H[:num_seeds] = 0.

        if self._solution_config.use_attention_aggregation:
            H_feat_dict = _attention_merge_feat(H_feat_dict, self.attention_model)
        elif self._solution_config.add_columns_dim_together:
            H_feat_dict = _cat_feat(H_feat_dict)
        else:
            H_feat_dict = _merge_feat(H_feat_dict)

        # For all situations, the embeddings after encoder can be kept here
        encoder_embeds = H_feat_dict[target_type]
        encoder_embeds = encoder_embeds[seed_lookup_idx[:, 0]]

        # Merge two dictionaries.
        H_node_dict = dict(H_id_dict)
        H_node_dict.update(H_feat_dict)
        for ntype in self.solution_config.embed_ntypes:
            if ntype in self.feat_encoder.node_out_size_dict:
                H_node_dict[ntype] = torch.cat(
                    [H_feat_dict[ntype], H_id_dict[ntype]], dim=1)

        # Encode edges.
        if self._solution_config.use_attention_aggregation:
            H_edge_dicts = [
                _attention_merge_feat(self.feat_encoder(edge_feat_dict), self.attention_model)
                for edge_feat_dict in edge_feat_dicts
            ]
        elif self._solution_config.add_columns_dim_together:
            H_edge_dicts = [
                _cat_feat(self.feat_encoder(edge_feat_dict))
                for edge_feat_dict in edge_feat_dicts
            ]
        else:
            H_edge_dicts = [
                _merge_feat(self.feat_encoder(edge_feat_dict))
                for edge_feat_dict in edge_feat_dicts
            ]

        # Message passing.
        H_node_dict = self.gnn(mfgs, H_node_dict, H_edge_dicts)

        # Prediction head.
        seed_embeds = self.seed_lookup_dict[task_name](H_node_dict, seed_lookup_idx)
        if self._solution_config.use_attention_aggregation:
            seed_ctx_embeds = _attention_merge_feat(
                self.feat_encoder(seed_feat_dict), self.attention_model
            )[f"__seed__-{task_name}"]
        elif self._solution_config.add_columns_dim_together:
            seed_ctx_embeds = (
                _cat_feat(self.feat_encoder(seed_feat_dict))[f"__seed__-{task_name}"]
            )
        else:
            seed_ctx_embeds = (
                _merge_feat(self.feat_encoder(seed_feat_dict))[f"__seed__-{task_name}"]
            )
        # return self.predictor_dict[task_name](seed_embeds, seed_ctx_embeds)

        src_embeds = seed_embeds[:, 0, :]
        dst_embeds = seed_embeds[:, 1, :]
        after_gnn_seed_embeds = src_embeds
        after_gnn_combine_embeds = torch.cat(
            [src_embeds + dst_embeds, src_embeds * dst_embeds, seed_ctx_embeds + dst_embeds, seed_ctx_embeds * dst_embeds],
            dim=1
        )

        return llm_embeds, encoder_embeds, after_gnn_seed_embeds, seed_ctx_embeds, after_gnn_combine_embeds

    def copy_inner_embeddings_to_model_device(self, model_device):
        # Only required for INVAR
        if hasattr(self.gnn, "copy_inner_embeddings_to_model_device"):
            self.gnn.copy_inner_embeddings_to_model_device(model_device)



class BaseGMLSolution(GraphMLSolution):
    """Base GML solution class."""

    def __init__(
        self,
        solution_config : BaseGNNSolutionConfig,
        data_config : GraphDatasetConfig
    ):
        # Set deterministic initialization
        torch.manual_seed(42)  # Set fixed seed
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        self.solution_config = solution_config
        self.data_config = data_config
        self.model = self.create_model()
        self._dataloaders = {}

        logger.debug(self.model)
        total_size = 0
        for param in self.model.parameters():
            total_size += param.nelement() * param.element_size()
        logger.debug(f"Model parameter size: {total_size/1024**2:.2f} MB")

    @abc.abstractmethod
    def create_model(self) -> nn.Module:
        pass

    def create_dataloader(
        self,
        item_set_dict : gb.ItemSetDict,
        graph : gb.sampling_graph.SamplingGraph,
        feat_store : gb.FeatureStore,
        device : DeviceInfo,
        mode : str,
        is_multi_gpu : bool = False,
        num_workers : Optional[int] = None
    ) -> DataLoader:
        model_device = 'cpu' if len(device.gpu_devices) == 0 else device.gpu_devices[0]
        if (id(item_set_dict), id(graph), id(feat_store), model_device, mode) not in self._dataloaders:
            self._dataloaders[id(item_set_dict), id(graph), id(feat_store), model_device, mode] = self._create_dataloader(
                item_set_dict, graph, feat_store, device, mode, is_multi_gpu=is_multi_gpu, num_workers=num_workers
            )
        return self._dataloaders[id(item_set_dict), id(graph), id(feat_store), model_device, mode]

    def _create_dataloader(
        self,
        item_set_dict : gb.ItemSetDict,
        graph : gb.sampling_graph.SamplingGraph,
        feat_store : gb.FeatureStore,
        device : DeviceInfo,
        mode : str,
        is_multi_gpu : bool = False,
        num_workers : Optional[int] = None
    ) -> DataLoader:
        shuffle = (mode == 'train')
        # shuffle = True
        batch_size = self.solution_config.batch_size \
            if mode == 'train' else self.solution_config.eval_batch_size
        # Declare data loading procedure.

        # 1. Sample items to form initial minibatch.
        if not is_multi_gpu:
            datapipe = gb.ItemSampler(
                item_set_dict, batch_size=batch_size, shuffle=shuffle)
        else:
            datapipe = gb.DistributedItemSampler(
                item_set=item_set_dict,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=False,
                # Some bugs exist, we have to drop uneven inputs even in validation.
                # But for testing, we still use single GPU and don't drop uneven inputs.
                drop_uneven_inputs=True,
            )

        # 2. (optional) negative sampling
        if self.data_config.task.task_type == DBBTaskType.retrieval and mode == 'train':
            datapipe = datapipe.tgif_sample_negative(
                graph,
                self.solution_config.negative_sampling_ratio,
                self.data_config.task.target_seed_idx,
                self.data_config.task.key_prediction_label_column,
                self.data_config.task.key_prediction_query_idx_column,
            )

        if mode == 'train' or self.solution_config.eval_fanouts is None:
            fanouts = self.solution_config.fanouts
        else:
            fanouts = self.solution_config.eval_fanouts
        if (
            self.solution_config.enable_temporal_sampling
            and self.data_config.task.seed_timestamp is not None
        ):
            logger.info("Using temporal neighbor sampler.")
            has_node_timestamp = TIMESTAMP_FEATURE_NAME in graph.node_attributes
            has_edge_timestamp = TIMESTAMP_FEATURE_NAME in graph.edge_attributes
            datapipe = datapipe.temporal_sample_neighbor(
                graph, fanouts=fanouts,
                node_timestamp_attr_name=TIMESTAMP_FEATURE_NAME if has_node_timestamp else None,
                edge_timestamp_attr_name=TIMESTAMP_FEATURE_NAME if has_edge_timestamp else None,
            )
        else:
            datapipe = datapipe.sample_neighbor(
                graph, fanouts=fanouts)

        # 4. (optional) exclude seed edges from the sampled subgraph.
        if self.data_config.task.seed_type in self.data_config.graph.etypes:
            datapipe = datapipe.transform(
                functools.partial(
                    gb.exclude_seed_edges, include_reverse_edges=True,
                    reverse_etypes_mapping=self.data_config.graph.reverse_etypes_mapping))

        # 5. Fetch node/edge features of the surrounding subgraph.
        node_feature_keys = {}
        for ntype, ft_cfg_dict in self.data_config.node_features.items():
            ft_names = [
                ft_name for ft_name, ft_cfg in ft_cfg_dict.items()
                if GraphFeatDictEncoder.is_valid_feat(ft_cfg)
            ]
            node_feature_keys[ntype] = ft_names
        edge_feature_keys = {}
        for etype, ft_cfg_dict in self.data_config.edge_features.items():
            ft_names = [
                ft_name for ft_name, ft_cfg in ft_cfg_dict.items()
                if GraphFeatDictEncoder.is_valid_feat(ft_cfg)
            ]
            edge_feature_keys[etype] = ft_names
        datapipe = datapipe.fetch_feature(
            feat_store, node_feature_keys, edge_feature_keys)

        # 6. Copy to training device.
        model_device = 'cpu' if len(device.gpu_devices) == 0 else device.gpu_devices[0]
        datapipe = datapipe.copy_to(model_device)

        # Create dataloader.
        if self.solution_config.use_multiprocessing:
            dataloader = gb.DataLoader(
                datapipe,
                num_workers=device.cpu_count // 2 if num_workers is None else num_workers
            )
        else:
            dataloader = gb.DataLoader(datapipe, num_workers=0)
        return dataloader

    def create_optimizer(self) -> torch.optim.Optimizer:
        return Adam(self.model.parameters(), lr=self.solution_config.lr)

    def get_input_node_ids(
        self,
        minibatch : gb.MiniBatch,
        device : DeviceInfo
    ) -> Dict[NType, torch.Tensor]:
        return {
            ntype : ids.to(device)
            for ntype, ids in minibatch.input_nodes.items()
        }

    def get_node_feats(
        self,
        minibatch : gb.MiniBatch,
    ) -> Dict[NType, Dict[str, torch.Tensor]]:
        node_feat_dict = defaultdict(dict)
        for (ntype, feat_name), feat in minibatch.node_features.items():
            node_feat_dict[ntype][feat_name] = feat
        return node_feat_dict

    def get_edge_feats(
        self, minibatch : gb.MiniBatch
    ) -> List[Dict[EType, Dict[str, torch.Tensor]]]:
        edge_feat_dicts = []
        for efeat_dict in minibatch.edge_features:
            new_efeat_dict = defaultdict(dict)
            for (etype, feat_name), feat in efeat_dict.items():
                new_efeat_dict[etype][feat_name] = feat
            edge_feat_dicts.append(new_efeat_dict)
        return edge_feat_dicts

    def get_seed_lookup_idx(
        self, minibatch : gb.MiniBatch
    ) -> Optional[torch.Tensor]:
        """Seed lookup index is used to lookup seed embeddings from
        the output of message passing, and arange them to align with
        the input seed tensor.

        For example, if the input seed shape is (N,K), the look up index
        is of the same shape (N,K). Suppose message passing computes
        node embedding of shape (M,D), where M is typically <= N*K because
        seeds can have duplicates. Then a lookup operation will gives
        a seed embedding tensor of shape (N,K,D).

        None return value means identity mapping.
        """
        if minibatch.seed_nodes is not None:
            return None
        else:
            assert minibatch.compacted_node_pairs is not None
            seed_type = self.data_config.task.seed_type
            if minibatch.compacted_negative_srcs is not None:
                pos_src, pos_dst = minibatch.compacted_node_pairs[seed_type]
                neg_src = minibatch.compacted_negative_srcs[seed_type]
                neg_dst = minibatch.compacted_negative_dsts[seed_type]
                # TODO(minjie): Here is another logic that is coupled tightly with GB's
                # internal behavior of how negative samples are aranged.
                neg_src = neg_src.view(-1, self.solution_config.negative_sampling_ratio)
                neg_dst = neg_dst.view(-1, self.solution_config.negative_sampling_ratio)
                all_src = torch.cat([pos_src.unsqueeze(1), neg_src], dim=1)
                all_dst = torch.cat([pos_dst.unsqueeze(1), neg_dst], dim=1)
                idx = torch.stack([all_src.reshape(-1), all_dst.reshape(-1)]).T
            else:
                pos_src, pos_dst = minibatch.compacted_node_pairs[seed_type]
                idx = torch.stack([pos_src, pos_dst]).T
            return idx

    def get_labels(
        self, minibatch : gb.MiniBatch
    ) -> torch.Tensor:
        seed_type = self.data_config.task.seed_type
        return minibatch.labels[seed_type]

    def get_seed_feats(
        self,
        minibatch : gb.MiniBatch,
        device : DeviceInfo,
        mode : str
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        seed_type = self.data_config.task.seed_type
        feats = {
            ft_name : getattr(minibatch, ft_name)[seed_type]
            for ft_name, ft_cfg in self.data_config.seed_features.items()
            if GraphFeatDictEncoder.is_valid_feat(ft_cfg)
        }
        if self.data_config.task.task_type == DBBTaskType.retrieval and mode == 'train':
            feats = {
                ft_name : feat[minibatch.query_idx[seed_type]]
                for ft_name, feat in feats.items()
            }
        feats = {ft_name : feat.to(device) for ft_name, feat in feats.items()}
        return {'__seed__' : feats}

    def get_query_idx(
        self,
        minibatch : gb.MiniBatch,
        device : DeviceInfo
    ) -> Optional[torch.Tensor]:
        if self.data_config.task.task_type == DBBTaskType.retrieval:
            query_idx = minibatch.query_idx[self.data_config.task.seed_type]
            return query_idx.to(device)
        else:
            return None

    @classmethod
    def fit_multi_gpu(cls, rank, solution_config, data_config, *args):
        instance = cls(solution_config, data_config)
        return instance._fit_multi_gpu_impl(rank, *args)

    def _fit_multi_gpu_impl(
        self,
        rank : int,
        dataset : DBBGraphDataset,
        task_name : str,
        ckpt_path : Path,
        device : DeviceInfo,
        world_size : int,
        enable_wandb : bool = True,
        shared_dict : Optional[Dict] = None,
        port : Optional[int] = None,
        wandb_run_id : Optional[str] = None
    ) -> FitSummary:
        ckpt_path = Path(ckpt_path)
        metric_fn = get_metric_fn(self.data_config.task)
        loss_fn = get_loss_fn(self.data_config.task)

        if rank == 0:
            wandb_config = {k : v for k, v in self.solution_config.__dict__.items()}
            wandb_config["solution"] = self.__class__.name
            wandb_config["dataset"] = dataset.dataset_name
            wandb_config["task"] = task_name
            wandb.init(
                project="Tab2graph",
                mode="online" if enable_wandb else "disabled",
                id=wandb_run_id,
                resume="must"
            )
            wandb.define_metric("val_metric", summary="max")

        device = DeviceInfo(
            gpu_devices=device.gpu_devices[rank:rank + 1],
            cpu_count=device.cpu_count
        )
        model_device = device.gpu_devices[0]
        torch.cuda.set_device(model_device)
        dist.init_process_group(
            backend="nccl",  # Use NCCL backend for distributed GPU training
            init_method="tcp://127.0.0.1:12345" if port is None else f"tcp://127.0.0.1:{port}",
            world_size=world_size,
            rank=rank,
        )
        model_this_rank = self.model.to(model_device)
        model_this_rank = DDP(
            model_this_rank,
            find_unused_parameters=True
        )

        train_loader = self.create_dataloader(
            dataset.graph_tasks[task_name].train_set,
            dataset.graph,
            dataset.feature,
            device,
            'train',
            is_multi_gpu=True
        )
        val_loader = self.create_dataloader(
            dataset.graph_tasks[task_name].validation_set,
            dataset.graph,
            dataset.feature,
            device,
            'eval',
            is_multi_gpu=True
        )
        optimizer = self.create_optimizer()
        num_batches = (
            (len(dataset.graph_tasks[task_name].train_set) // self.solution_config.batch_size + 1)
        ) // world_size

        best_val_metric = float('-inf')
        counter = 0
        if rank == 0:
            self.checkpoint(ckpt_path)
        with Join([model_this_rank]):
            for epoch in TimeBudgetedIterator(
                range(self.solution_config.epochs),
                self.solution_config.time_budget
            ):
                model_this_rank.train()
                total_loss = 0.
                total_train_metric = 0.
                tq = tqdm.tqdm(train_loader, total=num_batches) if rank == 0 else train_loader
                for step, minibatch in enumerate(tq):
                    # Prepare data.
                    input_node_id_dict = self.get_input_node_ids(minibatch, model_device)
                    node_feat_dict = self.get_node_feats(minibatch)
                    edge_feat_dicts = self.get_edge_feats(minibatch)
                    seed_feat_dict = self.get_seed_feats(minibatch, model_device, 'train')
                    seed_lookup_idx = self.get_seed_lookup_idx(minibatch)
                    query_idx = self.get_query_idx(minibatch, model_device)
                    labels = self.get_labels(minibatch)
                    mfgs = minibatch.blocks

                    # Forward.
                    logits = model_this_rank(
                        mfgs,
                        node_feat_dict,
                        input_node_id_dict,
                        edge_feat_dicts,
                        seed_feat_dict,
                        seed_lookup_idx
                    )
                    loss = loss_fn(logits, labels)
                    # Backward.
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Logging.
                    if step % 20 == 0:
                        # Metrics that are computationally costly.
                        if query_idx is not None:
                            query_idx = query_idx.cpu()
                        train_metric = metric_fn(
                            query_idx, logits.cpu(), labels.cpu()).item()
                        grad_norm = sum(
                            p.grad.norm() ** 2
                            for p in model_this_rank.parameters() if p.grad is not None
                        )
                    total_loss += loss.detach()
                    total_train_metric += train_metric
                    if rank == 0:
                        tq.set_postfix(
                            {
                                'Train loss': f'{loss.item():.4f}',
                                'Train metric': f'{train_metric:.4f}',
                                'Grad norm': f'{grad_norm:.6f}'
                            },
                            refresh=False,
                        )
                    if rank == 0:
                        wandb.log(
                            {'loss' : loss, 'grad_norm': grad_norm, 'train_metric': train_metric}
                        )

                total_loss /= step + 1
                total_train_metric /= step + 1

                # Evaluate the model on the validation set.
                model_this_rank.eval()
                val_num_batches = (
                    (len(dataset.graph_tasks[task_name].validation_set) // self.solution_config.eval_batch_size + 1)
                ) // world_size
                with torch.no_grad():
                    local_logits = []
                    local_labels = []
                    local_query_idx = []
                    tq = tqdm.tqdm(val_loader, total=val_num_batches) if rank == 0 else val_loader
                    for minibatch in tq:
                        # Prepare data.
                        input_node_id_dict = self.get_input_node_ids(minibatch, model_device)
                        node_feat_dict = self.get_node_feats(minibatch)
                        edge_feat_dicts = self.get_edge_feats(minibatch)
                        seed_feat_dict = self.get_seed_feats(minibatch, model_device, 'eval')
                        seed_lookup_idx = self.get_seed_lookup_idx(minibatch)
                        query_idx = self.get_query_idx(minibatch, model_device)
                        labels = self.get_labels(minibatch)
                        mfgs = minibatch.blocks

                        # Forward.
                        logits = model_this_rank(
                            mfgs,
                            node_feat_dict,
                            input_node_id_dict,
                            edge_feat_dicts,
                            seed_feat_dict,
                            seed_lookup_idx
                        )
                        local_logits.append(logits)
                        if query_idx is None:
                            local_query_idx = None
                        else:
                            local_query_idx.append(query_idx)
                        local_labels.append(labels)
                    # Wait for all ranks to finish evaluation.
                    local_logits = torch.cat(local_logits)
                    local_query_idx = torch.cat(local_query_idx) if local_query_idx is not None else None
                    local_labels = torch.cat(local_labels)

                    # Seems I can not use outer function here.
                    if rank == 0:
                        gather_logits = [torch.zeros_like(local_logits) for _ in range(world_size)]
                        gather_query_idx = [torch.zeros_like(local_query_idx) for _ in range(world_size)] if local_query_idx is not None else None
                        gather_labels = [torch.zeros_like(local_labels) for _ in range(world_size)]
                    else:
                        gather_logits = None
                        gather_query_idx = None
                        gather_labels = None
                    dist.gather(local_logits, gather_logits, dst=0)
                    if local_query_idx is not None:
                        dist.gather(local_query_idx, gather_query_idx, dst=0)
                    dist.gather(local_labels, gather_labels, dst=0)
                    if rank == 0:
                        final_logits = torch.cat(gather_logits).cpu()
                        final_query_idx = torch.cat(gather_query_idx).cpu() if gather_query_idx is not None else None
                        final_labels = torch.cat(gather_labels).cpu()
                        val_metric = metric_fn(final_query_idx, final_logits, final_labels).item()
                        if val_metric > best_val_metric:
                            logger.debug('Checkpointing ...')
                            best_val_metric = val_metric
                            counter = 0
                            self.checkpoint(ckpt_path)
                        else:
                            counter += 1
                            logger.info(
                                f"EarlyStopping counter: {counter} out of {self.solution_config.patience}"
                            )
                    # Spread the best_val_metric and counter to all ranks.
                    # Create tensors on the correct device
                    best_val_metric_tensor = torch.tensor(best_val_metric, device=model_device)
                    counter_tensor = torch.tensor(counter, device=model_device)

                    # Broadcast
                    dist.broadcast(best_val_metric_tensor, src=0)
                    dist.broadcast(counter_tensor, src=0)

                    # Update values from tensors
                    best_val_metric = best_val_metric_tensor.item()
                    counter = counter_tensor.item()

                    if counter >= self.solution_config.patience:
                        break
                dist.barrier()

                if rank == 0:
                    logger.info(
                        f"Epoch {epoch:04d} | loss: {total_loss:.4f} | "
                        f"train metric: {total_train_metric:.4f} | "
                        f"val metric: {val_metric:.4f} | "
                        f"best val metric: {best_val_metric:.4f}"
                    )
                    wandb.log({'val_metric' : val_metric})
                dist.barrier()

        if rank == 0:
            summary = FitSummary()
            summary.val_metric = float(best_val_metric)
            summary.train_metric = float(total_train_metric)
            shared_dict['fit_summary'] = summary

        return None

    def run(
        self,
        dataset : DBBGraphDataset,
        task_name : str,
        ckpt_path : Path,
        device : DeviceInfo,
        world_size : int,
        enable_wandb : bool = True,
        port : Optional[int] = None,
        wandb_run_id : Optional[str] = None
    ) -> FitSummary:
        # Thread limiting to avoid resource competition.
        os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count() // 2 // world_size)
        manager = mp.Manager()
        shared_dict = manager.dict()

        mp.set_sharing_strategy("file_system")
        mp.spawn(
            self.fit_multi_gpu,
            args=(
                self.solution_config,
                self.data_config,
                dataset,
                task_name,
                ckpt_path,
                device,
                world_size,
                enable_wandb,
                shared_dict,
                port,
                wandb_run_id
            ),
            nprocs=world_size,
            join=True,
        )

        return shared_dict.get('fit_summary', None)

    def fit(
        self,
        dataset : DBBGraphDataset,
        task_name : str,
        ckpt_path : Path,
        device : DeviceInfo
    ) -> FitSummary:
        ckpt_path = Path(ckpt_path)
        metric_fn = get_metric_fn(self.data_config.task)
        loss_fn = get_loss_fn(self.data_config.task)

        model_device = 'cpu' if len(device.gpu_devices) == 0 else device.gpu_devices[0]
        self.model = self.model.to(model_device)

        train_loader = self.create_dataloader(
            dataset.graph_tasks[task_name].train_set,
            dataset.graph, dataset.feature, device, 'train')
        # TODO(minjie): replace the following with len(train_loader) in the future.
        num_batches = len(dataset.graph_tasks[task_name].train_set) // self.solution_config.batch_size + 1

        optimizer = self.create_optimizer()

        best_val_metric = float('-inf')
        counter = 0
        self.checkpoint(ckpt_path)
        for epoch in TimeBudgetedIterator(
            range(self.solution_config.epochs),
            self.solution_config.time_budget
        ):
            self.model.train()
            total_loss = 0.
            total_train_metric = 0.
            with tqdm.tqdm(train_loader, total=num_batches) as tq:
                t0 = time.time()
                for step, minibatch in enumerate(tq):
                    # Prepare data.
                    input_node_id_dict = self.get_input_node_ids(minibatch, model_device)
                    node_feat_dict = self.get_node_feats(minibatch)
                    edge_feat_dicts = self.get_edge_feats(minibatch)
                    seed_feat_dict = self.get_seed_feats(minibatch, model_device, 'train')
                    seed_lookup_idx = self.get_seed_lookup_idx(minibatch)
                    query_idx = self.get_query_idx(minibatch, model_device)
                    labels = self.get_labels(minibatch)
                    mfgs = minibatch.blocks

                    # Forward.
                    logits = self.model(
                        mfgs,
                        node_feat_dict,
                        input_node_id_dict,
                        edge_feat_dicts,
                        seed_feat_dict,
                        seed_lookup_idx
                    )
                    loss = loss_fn(logits, labels)
                    # Backward.
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Logging.
                    if step % 20 == 0:
                        # Metrics that are computationally costly.
                        if query_idx is not None:
                            query_idx = query_idx.cpu()
                        train_metric = metric_fn(
                            query_idx, logits.cpu(), labels.cpu()).item()
                        grad_norm = sum(
                            p.grad.norm() ** 2
                            for p in self.model.parameters() if p.grad is not None
                        )
                    total_loss += loss.detach()
                    total_train_metric += train_metric
                    tq.set_postfix(
                        {
                            'Train loss': f'{loss.item():.4f}',
                            'Train metric': f'{train_metric:.4f}',
                            'Grad norm': f'{grad_norm:.6f}'
                        },
                        refresh=False,
                    )
                    wandb.log(
                        {'loss' : loss, 'grad_norm': grad_norm, 'train_metric': train_metric}
                    )

            total_loss /= step + 1
            total_train_metric /= step + 1

            val_metric = self.evaluate(
                dataset.graph_tasks[task_name].validation_set, dataset.graph, dataset.feature, device)
            if val_metric <= best_val_metric:
                counter += 1
                logger.info(
                    f"EarlyStopping counter: {counter} out of {self.solution_config.patience}"
                )
                if counter >= self.solution_config.patience:
                    break
            else:
                counter = 0
                logger.debug('Checkpointing ...')
                self.checkpoint(ckpt_path)
                best_val_metric = val_metric

            logger.info(
                f"Epoch {epoch:04d} | loss: {total_loss:.4f} | "
                f"train metric: {total_train_metric:.4f} | "
                f"val metric: {val_metric:.4f} | "
                f"best val metric: {best_val_metric:.4f}"
            )
            wandb.log({'val_metric' : val_metric})

        summary = FitSummary()
        summary.val_metric = float(best_val_metric)
        summary.train_metric = float(total_train_metric)

        return summary

    def evaluate(
        self,
        item_set_dict : gb.ItemSetDict,
        graph : gb.sampling_graph.SamplingGraph,
        feat_store : gb.FeatureStore,
        device : DeviceInfo,
        model_multi_gpu : nn.Module = None
    ) -> float:
        metric_fn = get_metric_fn(self.data_config.task)
        model_device = 'cpu' if len(device.gpu_devices) == 0 else device.gpu_devices[0]
        self.model = self.model.to(model_device)
        self.model.eval()

        eval_loader = self.create_dataloader(
            item_set_dict, graph, feat_store, device, 'eval')
        # TODO(minjie): replace the following with len(val_loader) in the future.
        num_batches = len(item_set_dict) // self.solution_config.eval_batch_size + 1
        with torch.no_grad():
            logits_per_trial = []
            labels_list = []
            query_idx_list = []
            for t in range(self.solution_config.eval_trials):
                logits_list = []
                for minibatch in tqdm.tqdm(eval_loader, total=num_batches):
                    # Prepare data.
                    input_node_id_dict = self.get_input_node_ids(minibatch, model_device)
                    node_feat_dict = self.get_node_feats(minibatch)
                    edge_feat_dicts = self.get_edge_feats(minibatch)
                    seed_feat_dict = self.get_seed_feats(minibatch, model_device, 'eval')
                    seed_lookup_idx = self.get_seed_lookup_idx(minibatch)
                    query_idx = self.get_query_idx(minibatch, model_device)
                    labels = self.get_labels(minibatch)
                    mfgs = minibatch.blocks

                    # Forward.
                    logits = self.model(
                        mfgs,
                        node_feat_dict,
                        input_node_id_dict,
                        edge_feat_dicts,
                        seed_feat_dict,
                        seed_lookup_idx
                    )
                    logits_list.append(logits)
                    if t == 0:
                        if query_idx is None:
                            query_idx_list = None
                        else:
                            query_idx_list.append(query_idx)
                        labels_list.append(labels)
                all_logits = torch.cat(logits_list)
                if t == 0:
                    if query_idx_list is None:
                        all_query_idx = None
                    else:
                        all_query_idx = torch.cat(query_idx_list).cpu()
                    all_labels = torch.cat(labels_list).cpu()
                logits_per_trial.append(all_logits.cpu())
        all_logits = torch.stack(logits_per_trial, 0).mean(0)
        return metric_fn(all_query_idx, all_logits, all_labels).item()

    def checkpoint(self, ckpt_path : Path) -> None:
        ckpt_path = Path(ckpt_path)
        torch.save(self.model.state_dict(), ckpt_path / 'model.pt')
        yaml_utils.save_pyd(self.solution_config, ckpt_path / 'solution_config.yaml')
        yaml_utils.save_pyd(self.data_config, ckpt_path / 'data_config.yaml')

    def load_from_checkpoint(self, ckpt_path : Path) -> None:
        ckpt_path = Path(ckpt_path)
        self.solution_config = yaml_utils.load_pyd(
            self.config_class, ckpt_path / 'solution_config.yaml')
        self.data_config = yaml_utils.load_pyd(
            GraphDatasetConfig, ckpt_path / 'data_config.yaml')
        self.model = self.create_model()
        self.model.load_state_dict(torch.load(ckpt_path / 'model.pt'))


def _cat_feat(feat_dict : Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    cat_feat_dict = {}
    for ty, ty_feat_dict in feat_dict.items():
        if len(ty_feat_dict) == 0:
            cat_feat_dict[ty] = None
        else:
            cat_feat_dict[ty] = torch.cat(
                [ty_feat_dict[feat_name] for feat_name in sorted(ty_feat_dict)], dim=1)
    return cat_feat_dict


def _merge_feat(feat_dict : Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    merged_feat_dict = {}
    for ty, ty_feat_dict in feat_dict.items():
        if len(ty_feat_dict) == 0:
            merged_feat_dict[ty] = None
        else:
            merged_feat_dict[ty] = torch.mean(
                torch.stack([ty_feat_dict[feat_name] for feat_name in sorted(ty_feat_dict)], dim=0), dim=0)

    return merged_feat_dict


def _attention_merge_feat(feat_dict : Dict[str, Dict[str, torch.Tensor]], task_feat, attention_model) -> Dict[str, torch.Tensor]:
    merged_feat_dict = {}
    for ty, ty_feat_dict in feat_dict.items():
        if len(ty_feat_dict) == 0:
            merged_feat_dict[ty] = None
        else:
            merged_feat_dict[ty] = []
            # convert the feat_dict to a list of tensors
            feat_list = [ty_feat_dict[feat_name] for feat_name in sorted(ty_feat_dict)]
            feat_list = [_ for _ in feat_list if _.shape[1]>1]
            # Turn the list of tensors to a tensor of shape (num_feats, num_nodes, feat_dim)
            #for feat_name in ty_feat_dict:
            #    print("feat_name", feat_name, ty_feat_dict[feat_name].shape, ty_feat_dict[feat_name].dtype) 
            feat_tensor = torch.stack(feat_list, dim=1)
            # Feed the tensor to the attention layer to get the merged tensor
            # The attention layer should be able to handle the shape (num_feats, num_nodes, feat_dim)
            # The output should be of shape (num_nodes, feat_dim)
            merged_feat_dict[ty] = attention_model(task_feat.reshape(1, 1, -1).expand(feat_tensor.shape[0], -1, -1), feat_tensor, None)

    return merged_feat_dict


def _attention_merge_feat_with_relation(
    feat_dict : Dict[str, Dict[str, torch.Tensor]],
    task_feat: torch.Tensor,
    data_config_node_features: Dict,
    attention_model,
    model_device: torch.device,
    return_relation_feat: bool = True
) -> Dict[str, torch.Tensor]:
    merged_feat_dict = {}
    relation_feat_dict = {}
    for ty, ty_feat_dict in feat_dict.items():
        if len(ty_feat_dict) == 0:
            merged_feat_dict[ty] = None
            relation_feat_dict[ty] = None
        else:
            merged_feat_dict[ty] = []
            # convert the feat_dict to a list of tensors
            feat_list = []
            column_name_emb_list = []
            for feat_name in sorted(ty_feat_dict):
                if feat_name.startswith("Griffin_text"):
                    feat_list.append(ty_feat_dict[feat_name])
                    # table can be data table or seed table. We check two cases.
                    # Column name can contain Griffin_text or not. We check two cases.
                    if ty.startswith("__seed__-"):
                        column_name = feat_name if feat_name in data_config_node_features[ty[9:]] else feat_name[13:]
                        column_name_emb_list.append(
                            torch.tensor(
                                data_config_node_features[ty[9:]][column_name].extra_fields['name_emb'],
                                device=model_device
                            )
                        )
                    else:
                        column_name = feat_name if feat_name in data_config_node_features[ty] else feat_name[13:]
                        column_name_emb_list.append(
                            torch.tensor(
                                data_config_node_features[ty][column_name].extra_fields['name_emb'],
                                device=model_device
                            )
                        )
            # Turn the list of tensors to a tensor of shape (num_feats, num_nodes, feat_dim)
            feat_tensor = torch.stack(feat_list, dim=1)
            column_name_emb_tensor = torch.stack(column_name_emb_list, dim=0)
            # Feed the tensor to the attention layer to get the merged tensor
            # The attention layer should be able to handle the shape (num_feats, num_nodes, feat_dim)
            # The output should be of shape (num_nodes, feat_dim)
            # Check if feat_tensor is empty
            if feat_tensor.shape[0] != 0:
                merged_feat_dict[ty] = attention_model(
                    task_feat.reshape(1, 1, -1).expand(feat_tensor.shape[0], -1, -1),
                    column_name_emb_tensor.reshape(
                        1, -1, column_name_emb_tensor.shape[-1]
                    ).expand(feat_tensor.shape[0], -1, -1),
                    feat_tensor,
                    None
                )
            if return_relation_feat:
                relation_feat_dict[ty] = {
                    ft_name: ty_feat_dict[ft_name]
                    for ft_name in sorted(ty_feat_dict)
                    if not ft_name.startswith("Griffin_text")
                }

    if return_relation_feat:
        return merged_feat_dict, relation_feat_dict
    else:
        return merged_feat_dict


class CombinedDataLoader_multi_gpu:
    def __init__(
        self,
        rank,
        task_name_list,
        # task_to_data_loader_dict,
        num_batches_dict=None,
        solution_configs=None,
        seed=42,
    ):
        # seed everything
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.left_task_name_list = task_name_list.copy()
        self.task_name_list = task_name_list.copy()
        # self.dataloader_params = task_to_data_loader_dict
        self.step_count = {task_name: 0 for task_name in task_name_list}
        self.left_num_batches_dict = num_batches_dict
        self.num_batches_dict = num_batches_dict.copy()
        self.shuffle_strategy = solution_configs.shuffle_strategy
        if self.shuffle_strategy == SHUFFLE_STRATEGY.RANDOM_TASK:
            self.task_name_list = random.sample(self.task_name_list, len(self.task_name_list))
        self.task_selector = random.Random(seed)
        self.seed = seed

    def reset(self):
        self.left_task_name_list = self.task_name_list.copy()
        self.left_num_batches_dict = self.num_batches_dict.copy()
        self.step_count = {task_name: 0 for task_name in self.task_name_list}

    def __iter__(self, dataloader_params):
        iterators = {
            task_name: iter(dataloader_params[task_name])
            for task_name in self.task_name_list
        }

        while len(self.left_task_name_list) > 0:
            task = self._sample_task()
            try:
                sample = next(iterators[task])
            except StopIteration:
                logger.info(f"StopIteration for task {task}")
                iterators[task] = iter(dataloader_params[task])
                sample = next(iterators[task])
            yield task, sample

    def _sample_task(self):
        if len(self.left_task_name_list) == 0:
            return None
        if self.shuffle_strategy == SHUFFLE_STRATEGY.DEFAULT or self.shuffle_strategy == SHUFFLE_STRATEGY.RANDOM_TASK:
            # Select the first task, update the left tasks
            task_name = self.left_task_name_list[0]
            self.step_count[task_name] += 1
            self.left_num_batches_dict[task_name] -= 1
            if self.left_num_batches_dict[task_name] == 0:
                self.left_task_name_list.remove(task_name)
            return task_name
        elif self.shuffle_strategy == SHUFFLE_STRATEGY.RANDOM_SAMPLE:
            # Randomly select one task
            assert np.sum(list(self.left_num_batches_dict.values())) > 0, "No tasks left to sample"
            task_name = self.task_selector.choices(
                population=self.left_task_name_list,
                weights=[
                    self.left_num_batches_dict[task_name]
                    for task_name in self.left_task_name_list
                ],
                k=1
            )[0]
            self.step_count[task_name] += 1
            self.left_num_batches_dict[task_name] -= 1
            if self.left_num_batches_dict[task_name] == 0:
                self.left_task_name_list.remove(task_name)
            return task_name
        else:
            raise ValueError(f"Invalid shuffle strategy: {self.shuffle_strategy}")


class BaseMultiTaskGMLSolution(GraphMLSolution):
    def __init__(
        self,
        solution_config : BaseGNNSolutionConfig,
        data_config : GraphDatasetConfig,
        provided_best_val_metric : Optional[float] = None
    ):
        # Set deterministic initialization
        torch.manual_seed(42)  # Set fixed seed
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        self.solution_config = solution_config
        self.data_config = data_config
        self.provided_best_val_metric = provided_best_val_metric
        self.model = self.create_model()
        self._dataloaders = {}

        logger.debug(self.model)
        total_size = 0
        for param in self.model.parameters():
            total_size += param.nelement() * param.element_size()
        logger.debug(f"Model parameter size: {total_size/1024**2:.2f} MB") 

    @abc.abstractmethod
    def create_model(self) -> nn.Module:
        pass

    @classmethod
    def fit_multi_gpu(cls, rank, solution_config, data_config, *args):
        instance = cls(solution_config, data_config)
        return instance._fit_multi_gpu_impl(rank, *args)

    def run(
        self,
        dataset : DBBGraphDataset,
        task_name_list : List[str],
        ckpt_path : Path,
        device : DeviceInfo,
        world_size : int,
        enable_wandb : bool = True,
        port : Optional[int] = None,
        wandb_run_id : Optional[str] = None
    ) -> FitSummary:
        # Thread limiting to avoid resource competition.
        os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count() // 2 // world_size)
        manager = mp.Manager()
        shared_dict = manager.dict()

        mp.set_sharing_strategy("file_system")
        mp.spawn(
            self.fit_multi_gpu,
            args=(
                self.solution_config,
                self.data_config,
                dataset,
                task_name_list,
                ckpt_path,
                device,
                world_size,
                enable_wandb,
                shared_dict,
                port,
                wandb_run_id
            ),
            nprocs=world_size,
            join=True,
        )

        return shared_dict.get('fit_summary', None)

    def evaluate_multi_task(
        self,
        item_set_dict : gb.ItemSetDict,
        graph : gb.sampling_graph.SamplingGraph,
        feat_store : gb.FeatureStore,
        device : DeviceInfo,
        task_name: str,
    ) -> float:
        task, seed_features = self.data_config.task_dict[task_name], self.data_config.seed_features_dict[task_name]
        metric_fn = get_metric_fn(task)
        model_device = 'cpu' if len(device.gpu_devices) == 0 else device.gpu_devices[0]
        self.model = self.model.to(model_device)
        self.model.eval()

        eval_loader = self.create_dataloader(
            item_set_dict, graph, feat_store, device, 'eval', task)
        # TODO(minjie): replace the following with len(val_loader) in the future.
        num_batches = len(item_set_dict) // self.solution_config.eval_batch_size + 1

        with torch.no_grad():
            logits_per_trial = []
            labels_list = []
            query_idx_list = []
            retrieve_target_labels_list = []
            for t in range(self.solution_config.eval_trials):
                logits_list = []
                for step, minibatch in enumerate(tqdm.tqdm(eval_loader, total=num_batches)):
                    # Prepare data.
                    input_node_id_dict = self.get_input_node_ids(minibatch, model_device)
                    node_feat_dict = self.get_node_feats(minibatch)
                    edge_feat_dicts = self.get_edge_feats(minibatch)
                    seed_feat_dict = self.get_seed_feats(minibatch, model_device, 'eval', task_name, task, seed_features)
                    seed_lookup_idx = self.get_seed_lookup_idx(minibatch, task)
                    query_idx = self.get_query_idx(minibatch, model_device, task)
                    labels = self.get_labels(minibatch, task)
                    mfgs = minibatch.blocks

                    # Forward.
                    logits = self.model(
                        mfgs,
                        node_feat_dict,
                        input_node_id_dict,
                        edge_feat_dicts,
                        seed_feat_dict,
                        seed_lookup_idx,
                        task_name,
                        task,
                        model_device
                    )
                    logits_list.append(logits)
                    if t == 0:
                        if query_idx is None:
                            query_idx_list = None
                        else:
                            query_idx_list.append(query_idx)
                            # retrieve_target_labels_list get the actual target labels for retrieval task.
                            # Will be used for computing roc_auc_score for retrieval task.
                            retrieve_target_labels_list.append(list(minibatch.node_pairs.values())[0][1].cpu())
                        labels_list.append(labels)
                if t == 0:
                    if query_idx_list is None:
                        all_query_idx = None
                    else:
                        all_query_idx = torch.cat(query_idx_list).cpu()
                        retrieve_target_labels_list = torch.cat(retrieve_target_labels_list).cpu()
                    all_labels = torch.cat(labels_list).cpu()
                all_logits = torch.cat(logits_list)
                logits_per_trial.append(all_logits.cpu())
        all_logits = torch.stack(logits_per_trial, 0).mean(0)
        return metric_fn(
            all_query_idx, all_logits, all_labels,
            retrieve_target_labels=retrieve_target_labels_list, eval_trials=self.solution_config.eval_trials
        ).item()

    def _fit_multi_gpu_impl(
        self,
        rank : int,
        dataset : DBBGraphDataset,
        task_name_list : List[str],
        ckpt_path : Path,
        device : DeviceInfo,
        world_size : int,
        enable_wandb : bool = True,
        shared_dict : Optional[Dict] = None,
        port : Optional[int] = None,
        wandb_run_id : Optional[str] = None
    ):
        ckpt_path = Path(ckpt_path)
        metric_fn_dict = {
            task_name : get_metric_fn(self.data_config.task_dict[task_name])
            for task_name in task_name_list
        }
        loss_fn_dict = {
            task_name : get_loss_fn(self.data_config.task_dict[task_name])
            for task_name in task_name_list
        }

        if rank == 0:
            wandb.init(
                project="Tab2graph",
                mode="online" if enable_wandb else "disabled",
                id=wandb_run_id,
                resume="must"
            )
            wandb.define_metric("val_metric", summary="max")

        device = DeviceInfo(
            gpu_devices=device.gpu_devices[rank:rank + 1],
            cpu_count=device.cpu_count
        )
        model_device = device.gpu_devices[0]
        torch.cuda.set_device(model_device)
        dist.init_process_group(
            backend="nccl",  # Use NCCL backend for distributed GPU training
            init_method="tcp://127.0.0.1:12345" if port is None else f"tcp://127.0.0.1:{port}",
            world_size=world_size,
            rank=rank,
        )
        model_this_rank = self.model.to(model_device)
        model_this_rank.copy_inner_embeddings_to_model_device(model_device)
        model_this_rank = DDP(
            model_this_rank,
            find_unused_parameters=True
        )
        train_loader_dict = {
            task_name : self.create_dataloader(
                dataset.graph_tasks[task_name].train_set,
                dataset.graph,
                dataset.feature,
                device,
                'train',
                self.data_config.task_dict[task_name],
                is_multi_gpu=True,
                num_workers=0,
                custom_batch_size=self.solution_config.batch_size_for_each_task.get(
                    task_name, None
                ),
                sample_ratio=self.solution_config.sample_ratio_for_each_task.get(
                    task_name, None
                )
            )
            for task_name in task_name_list
        }
        val_loader_dict = {
            task_name : self.create_dataloader(
                dataset.graph_tasks[task_name].validation_set,
                dataset.graph,
                dataset.feature,
                device,
                'eval',
                self.data_config.task_dict[task_name],
                is_multi_gpu=True,
                num_workers=0
            )
            for task_name in task_name_list
        }
        if self.solution_config.sample_ratio_for_each_task == {}:
            num_batches_dict = {
                task_name : len(dataset.graph_tasks[task_name].train_set) // self.solution_config.batch_size_for_each_task.get(
                    task_name, self.solution_config.batch_size
                )
                for task_name in task_name_list
            }
        else:
            num_batches_dict = {
                task_name : int(len(dataset.graph_tasks[task_name].train_set) * self.solution_config.sample_ratio_for_each_task.get(
                    task_name, 1.0
                )) // self.solution_config.batch_size_for_each_task.get(
                    task_name, self.solution_config.batch_size
                )
                for task_name in task_name_list
            }
        # Divide the total number of batches by world size
        num_batches_dict = {
            k : v // world_size
            for k, v in num_batches_dict.items()
        }
        total_num_batches = sum(num_batches_dict.values())

        optimizer = self.create_optimizer()
        optimizer_dict = {
            task_name : self.create_optimizer()
            for task_name in task_name_list
        }
        if self.solution_config.use_lr_scheduler:
            lr_scheduler_dict = {
                task_name : self.create_lr_scheduler(optimizer_dict[task_name])
                for task_name in task_name_list
            }

        if self.provided_best_val_metric is not None:
            best_val_metric = self.provided_best_val_metric
        else:
            best_val_metric = float('-inf')
        counter = 0
        if rank == 0:
            self.checkpoint(ckpt_path)
        train_metric_dict = {task_name: 0. for task_name in task_name_list}
        val_metric_history = {task_name: [] for task_name in task_name_list}

        combined_train_loader = CombinedDataLoader_multi_gpu(
            rank=rank,
            task_name_list=task_name_list,
            num_batches_dict=num_batches_dict,
            solution_configs=self.solution_config,
            seed=42
        )

        # Temporarily do not involve the dynamic sample and lr scheduler strategy in multi-gpu training.
        # TODO(yanbo): add it back.
        for epoch in TimeBudgetedIterator(
            range(self.solution_config.epochs),
            self.solution_config.time_budget
        ):
            combined_train_loader.reset()
            model_this_rank.train()
            grad_norm = {task_name: 0. for task_name in task_name_list}
            total_loss = 0.
            total_train_metric = 0.
            total_step = 0
            tq = (
                tqdm.tqdm(
                    combined_train_loader.__iter__(train_loader_dict),
                    total=total_num_batches,
                )
                if rank == 0
                else combined_train_loader.__iter__(train_loader_dict)
            )
            with Join([model_this_rank]):
                for step, minibatch_with_task in enumerate(tq):
                    task_name, minibatch = minibatch_with_task
                    loss_fn, metric_fn = loss_fn_dict[task_name], metric_fn_dict[task_name]
                    task, seed_features = self.data_config.task_dict[task_name], self.data_config.seed_features_dict[task_name]
                    input_node_id_dict = self.get_input_node_ids(minibatch, model_device)
                    node_feat_dict = self.get_node_feats(minibatch)
                    edge_feat_dicts = self.get_edge_feats(minibatch)
                    seed_feat_dict = self.get_seed_feats(minibatch, model_device, 'train', task_name, task, seed_features)
                    seed_lookup_idx = self.get_seed_lookup_idx(minibatch, task)
                    query_idx = self.get_query_idx(minibatch, model_device, task)
                    labels = self.get_labels(minibatch, task)
                    mfgs = minibatch.blocks

                    # Forward.
                    logits = self.model(
                        mfgs,
                        node_feat_dict,
                        input_node_id_dict,
                        edge_feat_dicts,
                        seed_feat_dict,
                        seed_lookup_idx,
                        task_name,
                        task,
                        model_device
                    )
                    loss = loss_fn(logits, labels)
                    # Backward.
                    optimizer = optimizer_dict[task_name]
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if (
                        rank == 0
                        and combined_train_loader.step_count[task_name] % 20 == 0
                        or combined_train_loader.step_count[task_name]
                        >= combined_train_loader.num_batches_dict[task_name]
                    ):
                        retrieve_target_labels = None
                        if query_idx is not None:
                            query_idx = query_idx.cpu()
                            retrieve_target_labels = list(minibatch.node_pairs.values())[0][1].cpu()
                        train_metric_dict[task_name] = metric_fn(
                            query_idx, logits.cpu(), labels.cpu(), retrieve_target_labels).item()
                        grad_norm[task_name] = sum(
                            p.grad.norm() ** 2
                            for p in self.model.parameters() if p.grad is not None
                        )
                    total_loss += loss.detach()
                    total_train_metric += train_metric_dict[task_name]
                    if rank == 0:
                        tq.set_postfix(
                            {
                                'Train loss': f'{loss.item():.4f}',
                                'Train metric': f'{train_metric_dict[task_name]:.4f}',
                                'Grad norm': f'{grad_norm[task_name]:.6f}'
                            },
                            refresh=False,
                        )

                        wandb.log(
                            {f'loss/{task_name}' : loss, f'grad_norm/{task_name}': grad_norm[task_name], f'train_metric/{task_name}': train_metric_dict[task_name]}
                        )
                    total_step += step + 1
            total_loss /= total_step
            total_train_metric /= total_step

            dist.barrier()
            model_this_rank.eval()
            val_metric_list = []
            for task_name in task_name_list:
                if rank == 0:
                    logger.info(f"Evaluating {task_name} ...")
                total_val_num_batches = len(
                    dataset.graph_tasks[task_name].validation_set
                ) // self.solution_config.eval_batch_size + (
                    1
                    if len(dataset.graph_tasks[task_name].validation_set)
                    % self.solution_config.eval_batch_size != 0
                    else 0
                )
                left_additional_batches = total_val_num_batches % world_size
                val_num_batches = total_val_num_batches // world_size + (1 if left_additional_batches > 0 else 0)
                task, seed_features = self.data_config.task_dict[task_name], self.data_config.seed_features_dict[task_name]
                metric_fn = metric_fn_dict[task_name]
                with torch.no_grad():
                    local_logits = []
                    local_labels = []
                    local_query_idx = []
                    local_retrieve_target_labels = []
                    tq = tqdm.tqdm(val_loader_dict[task_name], total=val_num_batches) if rank == 0 else val_loader_dict[task_name]
                    for minibatch in tq:
                        # Prepare data.
                        input_node_id_dict = self.get_input_node_ids(minibatch, model_device)
                        node_feat_dict = self.get_node_feats(minibatch)
                        edge_feat_dicts = self.get_edge_feats(minibatch)
                        seed_feat_dict = self.get_seed_feats(minibatch, model_device, 'eval', task_name, task, seed_features)
                        seed_lookup_idx = self.get_seed_lookup_idx(minibatch, task)
                        query_idx = self.get_query_idx(minibatch, model_device, task)
                        labels = self.get_labels(minibatch, task)
                        mfgs = minibatch.blocks

                        # Forward.
                        logits = model_this_rank(
                            mfgs,
                            node_feat_dict,
                            input_node_id_dict,
                            edge_feat_dicts,
                            seed_feat_dict,
                            seed_lookup_idx,
                            task_name,
                            task,
                            model_device
                        )
                        local_logits.append(logits)
                        if query_idx is None:
                            local_query_idx = None
                            local_retrieve_target_labels = None
                        else:
                            local_query_idx.append(query_idx)
                            local_retrieve_target_labels.append(
                                list(minibatch.node_pairs.values())[0][1]
                            )
                        local_labels.append(labels)
                    if rank >= left_additional_batches:
                        # Manually add barrier.
                        dist.barrier()
                        dist.barrier()
                    # Gather local_query_idx from rank 0.
                    # If the local_query_idx is not None, we need to gather it from all ranks.
                    # Otherwise, all query_idx are None.
                    if rank == 0:
                        query_idx_is_not_none = torch.tensor(local_query_idx is not None, device=model_device)
                        # Spread the query_idx_is_not_none to all ranks.
                        dist.broadcast(query_idx_is_not_none, src=0)
                    else:
                        query_idx_is_not_none = torch.tensor(False, device=model_device)
                        dist.broadcast(query_idx_is_not_none, src=0)
                    if len(local_logits) > 0:
                        local_logits = torch.cat(local_logits)
                        local_query_idx = torch.cat(
                            local_query_idx
                        ) if query_idx_is_not_none else None
                        local_labels = torch.cat(local_labels)
                        local_retrieve_target_labels = torch.cat(
                            local_retrieve_target_labels
                        ).to(model_device) if query_idx_is_not_none else None
                    else:
                        # For some processes, there might be no validation batches.
                        local_logits = torch.tensor([], dtype=torch.float, device=model_device)
                        local_query_idx = torch.tensor(
                            [], dtype=torch.long, device=model_device
                        ) if query_idx_is_not_none else None
                        local_labels = torch.tensor([], dtype=torch.long, device=model_device)
                        local_retrieve_target_labels = torch.tensor(
                            [], dtype=torch.long, device=model_device
                        ) if query_idx_is_not_none else None

                    gather_logits = gather_tensors(local_logits, rank, world_size)
                    if query_idx_is_not_none:
                        gather_query_idx = gather_tensors(local_query_idx, rank, world_size)
                    else:
                        gather_query_idx = None
                    gather_labels = gather_tensors(local_labels, rank, world_size)
                    if query_idx_is_not_none:
                        gather_retrieve_target_labels = gather_tensors(local_retrieve_target_labels, rank, world_size)
                    else:
                        gather_retrieve_target_labels = None
                    if rank == 0:
                        final_logits = torch.cat(gather_logits).cpu()
                        final_query_idx = torch.cat(
                            gather_query_idx
                        ).cpu() if query_idx_is_not_none else None
                        final_labels = torch.cat(gather_labels).cpu()
                        final_retrieve_target_labels = torch.cat(
                            gather_retrieve_target_labels
                        ).cpu() if query_idx_is_not_none else None
                        val_metric = metric_fn(
                            final_query_idx,
                            final_logits,
                            final_labels,
                            final_retrieve_target_labels
                        ).item()
                        val_metric_list.append(val_metric)

            if rank == 0:
                avg_val_metric = sum(val_metric_list) / len(val_metric_list)
                if avg_val_metric > best_val_metric:
                    logger.debug('Checkpointing ...')
                    best_val_metric = avg_val_metric
                    counter = 0
                    self.checkpoint(ckpt_path)
                else:
                    counter += 1
                    logger.info(
                        f"EarlyStopping counter: {counter} out of {self.solution_config.patience}"
                    )

            # Spread the best_val_metric and counter to all ranks.
            # Create tensors on the correct device
            best_val_metric_tensor = torch.tensor(best_val_metric, device=model_device)
            counter_tensor = torch.tensor(counter, device=model_device)

            # Broadcast
            dist.broadcast(best_val_metric_tensor, src=0)
            dist.broadcast(counter_tensor, src=0)
            # Update values from tensors
            best_val_metric = best_val_metric_tensor.item()
            counter = counter_tensor.item()

            if counter >= self.solution_config.patience:
                break
            dist.barrier()

            if rank == 0:
                logger.info(
                    f"Epoch {epoch:04d} | loss: {total_loss:.4f} | "
                    f"train metric: {total_train_metric:.4f} | "
                    f"val metric: {avg_val_metric:.4f} | "
                    f"best val metric: {best_val_metric:.4f}"
                )
                wandb.log({'val_metric' : avg_val_metric})
                for task_name, val_metric in zip(task_name_list, val_metric_list):
                    logger.info(f"{task_name} val metric: {val_metric:.4f}")
                    wandb.log({f'val_metric/{task_name}' : val_metric})

        if rank == 0:
            summary = FitSummary()
            summary.val_metric = float(best_val_metric)
            summary.train_metric = float(total_train_metric)
            shared_dict['fit_summary'] = summary

        return None

    def fit_multi_task_combined(
        self,
        dataset : DBBGraphDataset,
        task_name_list : List[str],
        test_task_name_list : List,
        ckpt_path : Path,
        device : DeviceInfo
    ) -> FitSummary:
        ckpt_path = Path(ckpt_path)
        metric_fn_dict = {
            task_name : get_metric_fn(self.data_config.task_dict[task_name])
            for task_name in task_name_list
        }
        loss_fn_dict = {
            task_name : get_loss_fn(self.data_config.task_dict[task_name])
            for task_name in task_name_list
        }

        model_device = 'cpu' if len(device.gpu_devices) == 0 else device.gpu_devices[0]
        self.model = self.model.to(model_device)

        random.seed(0)

        class CombinedDataLoader:
            def __init__(self, task_to_data_loader_dict, num_batches_dict=None, solution_configs=None):
                self.task_name_list = [task_name for task_name in task_to_data_loader_dict.keys()]
                self.dataloaders = task_to_data_loader_dict
                self.iterators = {
                    task_name : iter(dl)
                    for task_name, dl in task_to_data_loader_dict.items()
                }
                self.step_count = {task_name: 0 for task_name in task_name_list}
                self.num_batches_dict = num_batches_dict
                self.shuffle_strategy = solution_configs.shuffle_strategy
                if self.shuffle_strategy == "random_task":
                    self.task_name_list = random.sample(self.task_name_list, len(self.task_name_list))

            def __iter__(self):
                return self

            def __next__(self):
                if self.shuffle_strategy == SHUFFLE_STRATEGY.DEFAULT:
                    return self.default_next()
                elif self.shuffle_strategy == SHUFFLE_STRATEGY.RANDOM_TASK:
                    # default next and random task next are the same when iterating,
                    # the difference is in initialization of task orders
                    return self.default_next()
                elif self.shuffle_strategy == SHUFFLE_STRATEGY.RANDOM_SAMPLE:
                    return self.random_sample_next()
                else:
                    raise ValueError(f"Invalid shuffle strategy: {self.shuffle_strategy}")

            def default_next(self):
                if len(self.task_name_list) == 0:
                    del self.iterators
                    raise StopIteration
                task_name = self.task_name_list[0]
                it = self.iterators[task_name]
                try:
                    sample = next(it)
                except StopIteration:
                    self.iterators[task_name] = iter(self.dataloaders[task_name])
                    sample = next(self.iterators[task_name])

                self.step_count[task_name] += 1
                if self.step_count[task_name] >= self.num_batches_dict[task_name]:
                    self.task_name_list.remove(task_name)
                    self.iterators.pop(task_name)
                return task_name, sample

            def random_sample_next(self):
                # Randomly select one dataloader
                if len(self.task_name_list) == 0:
                    del self.iterators
                    raise StopIteration
                # the random probability is based on the number of batches left for each task
                task_name = random.choices(
                    population=self.task_name_list,
                    weights=[
                        self.num_batches_dict[task_name] - self.step_count[task_name]
                        for task_name in self.task_name_list
                    ],
                )[0]
                it = self.iterators[task_name]
                try:
                    sample = next(it)
                except StopIteration:
                    self.iterators[task_name] = iter(self.dataloaders[task_name])
                    sample = next(self.iterators[task_name])

                self.step_count[task_name] += 1
                if self.step_count[task_name] >= self.num_batches_dict[task_name]:
                    self.task_name_list.remove(task_name)
                    self.iterators.pop(task_name)
                return task_name, sample

        train_task_name_list = [task_name for task_name in task_name_list if task_name not in test_task_name_list]
        train_loader_dict = {
            task_name : self.create_dataloader(
                dataset.graph_tasks[task_name].train_set,
                dataset.graph, dataset.feature, device, 'train', self.data_config.task_dict[task_name])
            for task_name in train_task_name_list
        }
        raw_num_batches_dict = {
            task_name : len(dataset.graph_tasks[task_name].train_set) // self.solution_config.batch_size + 1
            for task_name in train_task_name_list
        }

        optimizer = self.create_optimizer()
        optimizer_dict = {
            task_name : self.create_optimizer()
            for task_name in train_task_name_list
        }
        if self.solution_config.use_lr_scheduler:
            lr_scheduler_dict = {
                task_name : self.create_lr_scheduler(optimizer_dict[task_name])
                for task_name in train_task_name_list
            }

        if self.provided_best_val_metric is not None:
            best_val_metric = self.provided_best_val_metric
        else:
            best_val_metric = float('-inf')
        counter = 0
        self.checkpoint(ckpt_path)
        train_metric_dict = {task_name: 0. for task_name in train_task_name_list}
        if self.solution_config.sample_ratio_for_each_task == {}:
            num_batches_dict = raw_num_batches_dict
        else:
            logger.info("Using sample_ratio_for_each_task")
            num_batches_dict = {
                task_name : int(
                    raw_num_batches_dict[task_name] * self.solution_config.sample_ratio_for_each_task.get(task_name, 1.0)
                )
                for task_name in train_task_name_list
            }
        total_num_batches = sum(num_batches_dict.values())
        val_metric_history = {task_name: [] for task_name in train_task_name_list}

        for epoch in TimeBudgetedIterator(
            range(self.solution_config.epochs),
            self.solution_config.time_budget
        ):
            if self.solution_config.dynamic_sample_strategy:
                (
                    epoch_num_per_update,
                    update_ratio,
                    update_percentage,
                    clip_ratio_lowerbound,
                    clip_ratio_upperbound,
                    target_metrics_dict,
                ) = parse_dynamic_sample_strategy(
                    self.solution_config.dynamic_sample_strategy
                )
                if epoch < 2 * epoch_num_per_update:
                    pass
                elif epoch % epoch_num_per_update == 0:
                    cur_val_metrics = {
                        task_name: sum(
                            val_metric_history[task_name][-epoch_num_per_update:]
                        )
                        / epoch_num_per_update
                        for task_name in train_task_name_list
                    }
                    last_val_metrics = {
                        task_name: sum(
                            val_metric_history[task_name][
                                -2 * epoch_num_per_update : -epoch_num_per_update
                            ]
                        )
                        / epoch_num_per_update
                        for task_name in train_task_name_list
                    }
                    converged_task_name_list = [
                        task_name
                        for task_name in train_task_name_list
                        if cur_val_metrics[task_name] >= target_metrics_dict[task_name]
                    ]
                    left_task_name_list = [
                        task_name
                        for task_name in train_task_name_list
                        if task_name not in converged_task_name_list
                    ]
                    improve_ratio = {
                        task_name: (
                            cur_val_metrics[task_name] - last_val_metrics[task_name]
                        )
                        / (target_metrics_dict[task_name] - last_val_metrics[task_name])
                        for task_name in left_task_name_list
                    }
                    # get improve_ratio in top update_percentage tasks
                    improve_ratio_top = sorted(
                        improve_ratio.items(), key=lambda x: x[1], reverse=True
                    )[: int(update_percentage * len(left_task_name_list))]
                    improve_ratio_bottom = sorted(
                        improve_ratio.items(), key=lambda x: x[1], reverse=False
                    )[: int(update_percentage * len(left_task_name_list))]
                    if improve_ratio_top:
                        for task_name, _ in improve_ratio_top:
                            num_batches_dict[task_name] = int(
                                num_batches_dict[task_name] * (1 - update_ratio)
                            )
                    if improve_ratio_bottom:
                        for task_name, _ in improve_ratio_bottom:
                            num_batches_dict[task_name] = int(
                                num_batches_dict[task_name] * (1 + update_ratio)
                            )
                    if converged_task_name_list:
                        for task_name in converged_task_name_list:
                            num_batches_dict[task_name] = int(
                                raw_num_batches_dict[task_name] * self.solution_config.sample_ratio_for_each_task[task_name] * clip_ratio_lowerbound
                            )

                    # clip num_batches_dict to [clip_ratio_lowerbound, clip_ratio_upperbound]
                    for task_name in train_task_name_list:
                        num_batches_dict[task_name] = max(
                            num_batches_dict[task_name],
                            int(
                                raw_num_batches_dict[task_name] * self.solution_config.sample_ratio_for_each_task[task_name] * clip_ratio_lowerbound
                            ),
                        )
                        num_batches_dict[task_name] = min(
                            num_batches_dict[task_name],
                            int(
                                raw_num_batches_dict[task_name] * self.solution_config.sample_ratio_for_each_task[task_name] * clip_ratio_upperbound
                            ),
                        )
                total_num_batches = sum(num_batches_dict.values())
                for task_name in train_task_name_list:
                    logger.info(
                        f"Epoch {epoch:04d} | Task {task_name} | num_batches: {num_batches_dict[task_name]}"
                    )
                    wandb.log({f'num_batches/{task_name}': num_batches_dict[task_name]})
            self.model.train()
            grad_norm = {task_name: 0. for task_name in train_task_name_list}
            total_loss = 0.
            total_train_metric = 0.
            total_step = 0
            combined_train_loader = CombinedDataLoader(train_loader_dict, num_batches_dict, self.solution_config)
            with tqdm.tqdm(combined_train_loader, total=total_num_batches) as tq:
                for step, minibatch_with_task in enumerate(tq):
                    task_name, minibatch = minibatch_with_task
                    loss_fn, metric_fn = loss_fn_dict[task_name], metric_fn_dict[task_name]
                    task, seed_features = self.data_config.task_dict[task_name], self.data_config.seed_features_dict[task_name]
                    input_node_id_dict = self.get_input_node_ids(minibatch, model_device)
                    node_feat_dict = self.get_node_feats(minibatch)
                    edge_feat_dicts = self.get_edge_feats(minibatch)
                    seed_feat_dict = self.get_seed_feats(minibatch, model_device, 'train', task_name, task, seed_features)
                    seed_lookup_idx = self.get_seed_lookup_idx(minibatch, task)
                    query_idx = self.get_query_idx(minibatch, model_device, task)
                    labels = self.get_labels(minibatch, task)
                    mfgs = minibatch.blocks

                    # Forward.
                    logits = self.model(
                        mfgs,
                        node_feat_dict,
                        input_node_id_dict,
                        edge_feat_dicts,
                        seed_feat_dict,
                        seed_lookup_idx,
                        task_name,
                        task,
                        model_device
                    )
                    loss = loss_fn(logits, labels)
                    # Backward.
                    optimizer = optimizer_dict[task_name]
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if (
                        combined_train_loader.step_count[task_name] % 20 == 0
                        or combined_train_loader.step_count[task_name]
                        >= combined_train_loader.num_batches_dict[task_name]
                    ):
                        retrieve_target_labels = None
                        if query_idx is not None:
                            query_idx = query_idx.cpu()
                            retrieve_target_labels = list(minibatch.node_pairs.values())[0][1].cpu()
                        train_metric_dict[task_name] = metric_fn(
                            query_idx, logits.cpu(), labels.cpu(), retrieve_target_labels).item()
                        grad_norm[task_name] = sum(
                            p.grad.norm() ** 2
                            for p in self.model.parameters() if p.grad is not None
                        )
                    total_loss += loss.detach()
                    total_train_metric += train_metric_dict[task_name]
                    tq.set_postfix(
                        {
                            'Train loss': f'{loss.item():.4f}',
                            'Train metric': f'{train_metric_dict[task_name]:.4f}',
                            'Grad norm': f'{grad_norm[task_name]:.6f}'
                        },
                        refresh=False,
                    )

                    wandb.log(
                        {f'loss/{task_name}' : loss, f'grad_norm/{task_name}': grad_norm[task_name], f'train_metric/{task_name}': train_metric_dict[task_name]}
                    )
                total_step += step + 1
            total_loss /= total_step
            total_train_metric /= total_step

            train_val_metric_list = [
                self.evaluate_multi_task(
                    dataset.graph_tasks[task_name].validation_set, dataset.graph, dataset.feature, device, task_name)
                for task_name in train_task_name_list
            ]
            for task_name, val_metric in zip(train_task_name_list, train_val_metric_list):
                val_metric_history[task_name].append(val_metric)
            if test_task_name_list:
                test_val_metric_list = [
                    self.evaluate_multi_task(
                        dataset.graph_tasks[task_name].validation_set,
                        dataset.graph, dataset.feature,
                        device,
                        task_name,
                    )
                    for task_name in test_task_name_list
                ]
            else:
                test_val_metric_list = []
            # avg_val_metric only includes the tasks that are not in test_task_name_list
            avg_val_metric = sum(train_val_metric_list) / len(train_val_metric_list)
            if avg_val_metric <= best_val_metric:
                counter += 1
                logger.info(
                    f"EarlyStopping counter: {counter} out of {self.solution_config.patience}"
                )
                if counter >= self.solution_config.patience:
                    break
            else:
                counter = 0
                logger.debug('Checkpointing ...')
                self.checkpoint(ckpt_path)
                best_val_metric = avg_val_metric
            if self.solution_config.use_lr_scheduler:
                for task_name, val_metric in zip(train_task_name_list, train_val_metric_list):
                    # update learning rate
                    # if learning rate is updated, log the new learning rate
                    if lr_scheduler_dict[task_name].step(val_metric):
                        logger.info(f"Learning rate of {task_name} is updated to {optimizer_dict[task_name].param_groups[0]['lr']}")
                    wandb.log({f'lr/{task_name}': optimizer_dict[task_name].param_groups[0]['lr']})

            logger.info(
                f"Epoch {epoch:04d} | loss: {total_loss:.4f} | "
                f"train metric: {total_train_metric:.4f} | "
                f"val metric: {avg_val_metric:.4f} | "
                f"best val metric: {best_val_metric:.4f}"
            )
            wandb.log({'avg_val_metric' : avg_val_metric})
            for task_name, val_metric in zip(train_task_name_list, train_val_metric_list):
                logger.info(f"{task_name} val metric: {val_metric:.4f}")
                wandb.log({f'val_metric/{task_name}' : val_metric})
            if test_task_name_list:
                for task_name, val_metric in zip(test_task_name_list, test_val_metric_list):
                    for evaluation_setting in val_metric.keys():
                        logger.info(f"{task_name} val metric ({evaluation_setting}-shot): {val_metric[evaluation_setting]:.4f}")
                        wandb.log({f'val_metric/{task_name}/{evaluation_setting}-shot' : val_metric[evaluation_setting]})

        summary = FitSummary()
        summary.val_metric = float(best_val_metric)
        summary.train_metric = float(total_train_metric)

        return summary

    def create_dataloader(
        self,
        item_set_dict : gb.ItemSetDict,
        graph : gb.sampling_graph.SamplingGraph,
        feat_store : gb.FeatureStore,
        device : DeviceInfo,
        mode : str,
        task,
        is_multi_gpu : bool = False,
        num_workers : Optional[int] = None,
        custom_batch_size : Optional[int] = None,
        sample_ratio : Optional[float] = None
    ) -> DataLoader:
        model_device = 'cpu' if len(device.gpu_devices) == 0 else device.gpu_devices[0]
        if (id(item_set_dict), id(graph), id(feat_store), model_device, mode) not in self._dataloaders:
            self._dataloaders[id(item_set_dict), id(graph), id(feat_store), model_device, mode] = self._create_dataloader(
                item_set_dict,
                graph,
                feat_store,
                device,
                mode,
                task,
                is_multi_gpu,
                num_workers,
                custom_batch_size,
                sample_ratio
            )
        return self._dataloaders[id(item_set_dict), id(graph), id(feat_store), model_device, mode]

    def _create_dataloader(
        self,
        item_set_dict : gb.ItemSetDict,
        graph : gb.sampling_graph.SamplingGraph,
        feat_store : gb.FeatureStore,
        device : DeviceInfo,
        mode : str,
        task,
        is_multi_gpu : bool = False,
        num_workers : Optional[int] = None,
        custom_batch_size : Optional[int] = None,
        sample_ratio : Optional[float] = None
    ) -> DataLoader:
        shuffle = (mode == 'train')
        # For retrieval tasks, we should shuffle to avoid ties.
        # shuffle = True
        batch_size = custom_batch_size \
            if custom_batch_size is not None else \
            self.solution_config.batch_size if mode == 'train' else self.solution_config.eval_batch_size
        # if sample_ratio is not None, we need to sample the items
        # The operation is done on item_set_dict
        if sample_ratio is not None:
            assert len(item_set_dict._itemsets) == 1, "Only support one item set for now"
            original_length = len(item_set_dict)
            new_length = int(original_length * sample_ratio)
            # generate a random index list of length new_length
            random_index_list = torch.randint(0, original_length, (new_length,))
            [(key, value)] = item_set_dict._itemsets.items()
            new_items = tuple(
                item[random_index_list] for item in value._items
            )
            new_item_set = gb.ItemSet(new_items, names=item_set_dict._names)
            item_set_dict = gb.ItemSetDict({key : new_item_set})
        # 1. Sample items to form initial minibatch.
        if not is_multi_gpu:
            datapipe = gb.ItemSampler(
                item_set_dict, batch_size=batch_size, shuffle=shuffle)
        else:
            datapipe = gb.DistributedItemSampler(
                item_set=item_set_dict,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=False,
                drop_uneven_inputs=(mode == 'train'),
            )

        # 2. (optional) negative sampling
        if task.task_type == DBBTaskType.retrieval and mode == 'train':
            datapipe = datapipe.tgif_sample_negative(
                graph,
                self.solution_config.negative_sampling_ratio,
                task.target_seed_idx,
                task.key_prediction_label_column,
                task.key_prediction_query_idx_column,
                strict_mode=self.solution_config.strict_mode,
                # For Griffin classification task, we have the num_classes, so we can use it.
                # For Griffin link prediction task, we do not have the num_classes, so we do not use it.
                num_classes=task.num_classes if hasattr(task, 'num_classes') else None,
            )

        if mode == 'train' or self.solution_config.eval_fanouts is None:
            fanouts = self.solution_config.fanouts
        else:
            fanouts = self.solution_config.eval_fanouts
        if (
            self.solution_config.enable_temporal_sampling
            and task.seed_timestamp is not None
        ):
            logger.info("Using temporal neighbor sampler.")
            has_node_timestamp = TIMESTAMP_FEATURE_NAME in graph.node_attributes
            has_edge_timestamp = TIMESTAMP_FEATURE_NAME in graph.edge_attributes
            datapipe = datapipe.temporal_sample_neighbor(
                graph, fanouts=fanouts,
                node_timestamp_attr_name=TIMESTAMP_FEATURE_NAME if has_node_timestamp else None,
                edge_timestamp_attr_name=TIMESTAMP_FEATURE_NAME if has_edge_timestamp else None,
            )
        else:
            datapipe = datapipe.sample_neighbor(
                graph, fanouts=fanouts)

        # 4. (optional) exclude seed edges from the sampled subgraph.
        #TODO: Consider more situations.
        if task.task_type == DBBTaskType.retrieval:
            seed_etuple = task.seed_type.split(':')
            if seed_etuple[1].startswith('reverse_'):
                rev_type = seed_etuple[1][8:]
            else:
                rev_type = 'reverse_' + seed_etuple[1]
            reverse_seed_type = f"{seed_etuple[2]}:{rev_type}:{seed_etuple[0]}"
            if task.seed_type in self.data_config.graph.etypes:
                datapipe = datapipe.transform(
                    functools.partial(
                        gb.exclude_seed_edges, include_reverse_edges=True,
                        reverse_etypes_mapping=self.data_config.graph.reverse_etypes_mapping))
            elif reverse_seed_type in self.data_config.graph.etypes:
                datapipe = datapipe.transform(
                    functools.partial(
                        gb.exclude_seed_edges, include_reverse_edges=True,
                        reverse_etypes_mapping=self.data_config.graph.reverse_etypes_mapping))

        # 5. Fetch node/edge features of the surrounding subgraph.
        node_feature_keys = {}
        for ntype, ft_cfg_dict in self.data_config.node_features.items():
            ft_names = [
                ft_name for ft_name, ft_cfg in ft_cfg_dict.items()
                if GraphFeatDictEncoder.is_valid_feat(ft_cfg)
            ]
            node_feature_keys[ntype] = ft_names
        edge_feature_keys = {}
        for etype, ft_cfg_dict in self.data_config.edge_features.items():
            ft_names = [
                ft_name for ft_name, ft_cfg in ft_cfg_dict.items()
                if GraphFeatDictEncoder.is_valid_feat(ft_cfg)
            ]
            edge_feature_keys[etype] = ft_names
        datapipe = datapipe.fetch_feature(
            feat_store, node_feature_keys, edge_feature_keys)

        # 6. Copy to training device.
        model_device = 'cpu' if len(device.gpu_devices) == 0 else device.gpu_devices[0]
        datapipe = datapipe.copy_to(model_device)

        # Create dataloader.
        if self.solution_config.use_multiprocessing:
            dataloader = gb.DataLoader(
                datapipe,
                num_workers=device.cpu_count // 2 if not is_multi_gpu else num_workers
            )
        else:
            dataloader = gb.DataLoader(datapipe, num_workers=0)
        return dataloader

    def create_optimizer(self) -> torch.optim.Optimizer:
        return Adam(self.model.parameters(), lr=self.solution_config.lr)

    def create_lr_scheduler(self, optimizer : torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=self.solution_config.lr_scheduler_patience,
            verbose=True
        )

    def get_input_node_ids(
        self,
        minibatch : gb.MiniBatch,
        device : DeviceInfo
    ) -> Dict[NType, torch.Tensor]:
        return {
            ntype : ids.to(device)
            for ntype, ids in minibatch.input_nodes.items()
        }

    def get_node_feats(
        self,
        minibatch : gb.MiniBatch,
    ) -> Dict[NType, Dict[str, torch.Tensor]]:
        node_feat_dict = defaultdict(dict)
        for (ntype, feat_name), feat in minibatch.node_features.items():
            node_feat_dict[ntype][feat_name] = feat
        return node_feat_dict

    def get_edge_feats(
        self, minibatch : gb.MiniBatch
    ) -> List[Dict[EType, Dict[str, torch.Tensor]]]:
        edge_feat_dicts = []
        for efeat_dict in minibatch.edge_features:
            new_efeat_dict = defaultdict(dict)
            for (etype, feat_name), feat in efeat_dict.items():
                new_efeat_dict[etype][feat_name] = feat
            edge_feat_dicts.append(new_efeat_dict)
        return edge_feat_dicts

    def get_seed_lookup_idx(
        self, minibatch : gb.MiniBatch, task
    ) -> Optional[torch.Tensor]:
        """Seed lookup index is used to lookup seed embeddings from
        the output of message passing, and arange them to align with
        the input seed tensor.

        For example, if the input seed shape is (N,K), the look up index
        is of the same shape (N,K). Suppose message passing computes
        node embedding of shape (M,D), where M is typically <= N*K because
        seeds can have duplicates. Then a lookup operation will gives
        a seed embedding tensor of shape (N,K,D).

        None return value means identity mapping.
        """
        if minibatch.seed_nodes is not None:
            return None
        else:
            assert minibatch.compacted_node_pairs is not None
            seed_type = task.seed_type
            if minibatch.compacted_negative_srcs is not None:
                pos_src, pos_dst = minibatch.compacted_node_pairs[seed_type]
                neg_src = minibatch.compacted_negative_srcs[seed_type]
                neg_dst = minibatch.compacted_negative_dsts[seed_type]
                # TODO(minjie): Here is another logic that is coupled tightly with GB's
                # internal behavior of how negative samples are aranged.
                if self.solution_config.strict_mode == StrictType.full and hasattr(task, 'num_classes'):
                    # For Griffin classification task
                    neg_src = neg_src.view(-1, task.num_classes - 1)
                    neg_dst = neg_dst.view(-1, task.num_classes - 1)
                else:
                    neg_src = neg_src.view(-1, self.solution_config.negative_sampling_ratio)
                    neg_dst = neg_dst.view(-1, self.solution_config.negative_sampling_ratio)
                all_src = torch.cat([pos_src.unsqueeze(1), neg_src], dim=1)
                all_dst = torch.cat([pos_dst.unsqueeze(1), neg_dst], dim=1)
                idx = torch.stack([all_src.reshape(-1), all_dst.reshape(-1)]).T
            else:
                pos_src, pos_dst = minibatch.compacted_node_pairs[seed_type]
                idx = torch.stack([pos_src, pos_dst]).T
            return idx

    def get_seed_lookup_idx_with_only_src(
        self, minibatch : gb.MiniBatch, task
    ) -> Optional[torch.Tensor]:
        """Seed lookup index is used to lookup seed embeddings from
        the output of message passing, and arange them to align with
        the input seed tensor.

        For example, if the input seed shape is (N,K), the look up index
        is of the same shape (N,K). Suppose message passing computes
        node embedding of shape (M,D), where M is typically <= N*K because
        seeds can have duplicates. Then a lookup operation will gives
        a seed embedding tensor of shape (N,K,D).

        None return value means identity mapping.
        """
        if minibatch.seed_nodes is not None:
            return None
        else:
            assert minibatch.compacted_node_pairs is not None
            seed_type = task.seed_type
            if minibatch.compacted_negative_srcs is not None:
                pos_src, pos_dst = minibatch.compacted_node_pairs[seed_type]
                neg_src = minibatch.compacted_negative_srcs[seed_type]
                neg_dst = minibatch.compacted_negative_dsts[seed_type]
                # TODO(minjie): Here is another logic that is coupled tightly with GB's
                # internal behavior of how negative samples are aranged.
                if self.solution_config.strict_mode == StrictType.full:
                    neg_src = neg_src.view(-1, task.num_classes - 1)
                    neg_dst = neg_dst.view(-1, task.num_classes - 1)
                else:
                    neg_src = neg_src.view(-1, self.solution_config.negative_sampling_ratio)
                    neg_dst = neg_dst.view(-1, self.solution_config.negative_sampling_ratio)
                # all_src = torch.cat([pos_src.unsqueeze(1), neg_src], dim=1)
                # all_dst = torch.cat([pos_dst.unsqueeze(1), neg_dst], dim=1)
                # idx = torch.stack([all_src.reshape(-1), all_dst.reshape(-1)]).T
                # only the pos_src is needed
                idx = pos_src.unsqueeze(1)
            else:
                pos_src, pos_dst = minibatch.compacted_node_pairs[seed_type]
                # idx = torch.stack([pos_src, pos_dst]).T
                idx = pos_src.unsqueeze(1)
            return idx

    def get_labels(
        self, minibatch : gb.MiniBatch, task
    ) -> torch.Tensor:
        seed_type = task.seed_type
        return minibatch.labels[seed_type]

    def get_seed_feats(
        self,
        minibatch : gb.MiniBatch,
        device : DeviceInfo,
        mode : str,
        task_name : str,
        task,
        seed_features
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        seed_type = task.seed_type
        feats = {
            ft_name : getattr(minibatch, ft_name)[seed_type]
            for ft_name, ft_cfg in seed_features.items()
            if GraphFeatDictEncoder.is_valid_feat(ft_cfg)
        }
        if task.task_type == DBBTaskType.retrieval and mode == 'train':
            feats = {
                ft_name : feat[minibatch.query_idx[seed_type]]
                for ft_name, feat in feats.items()
            }
        feats = {ft_name : feat.to(device) for ft_name, feat in feats.items()}
        return {f'__seed__-{task_name}' : feats}

    def get_query_idx(
        self,
        minibatch : gb.MiniBatch,
        device : DeviceInfo,
        task
    ) -> Optional[torch.Tensor]:
        if task.task_type == DBBTaskType.retrieval:
            query_idx = minibatch.query_idx[task.seed_type]
            return query_idx.to(device)
        else:
            return None

    def checkpoint(self, ckpt_path : Path) -> None:
        ckpt_path = Path(ckpt_path)
        torch.save(self.model.state_dict(), ckpt_path / 'model.pt')
        yaml_utils.save_pyd(self.solution_config, ckpt_path / 'solution_config.yaml')
        yaml_utils.save_pyd(self.data_config, ckpt_path / 'data_config.yaml')

    def load_from_checkpoint(self, ckpt_path : Path) -> None:
        ckpt_path = Path(ckpt_path)
        self.solution_config = yaml_utils.load_pyd(
            self.config_class, ckpt_path / 'solution_config.yaml')
        self.data_config = yaml_utils.load_pyd(
            GraphDatasetMultiTaskConfig, ckpt_path / 'data_config.yaml')
        self.model = self.create_model()
        self.model.load_state_dict(torch.load(ckpt_path / 'model.pt'))


def gather_tensors(tensor, rank, world_size):
    # Step 1: Gather shape of tensors
    device = tensor.device
    local_shape = torch.tensor(tensor.shape, dtype=torch.long).to(device)
    if rank == 0:
        gather_shapes = [torch.zeros_like(local_shape) for _ in range(world_size)]
    else:
        gather_shapes = None
    dist.gather(local_shape, gather_shapes, dst=0)

    # Step 2: Gather tensors
    if rank == 0:
        max_size = max(size.item() for size in gather_shapes)
        gather_tensors = [
            torch.zeros(max_size, dtype=tensor.dtype, device=device) for _ in range(world_size)
        ]
    else:
        max_size = 0
        gather_tensors = None
    max_size = torch.tensor(max_size, dtype=torch.long, device=device)
    dist.broadcast(max_size, src=0)

    padded_tensor = torch.cat(
        [tensor, torch.zeros(max_size - tensor.shape[0], dtype=tensor.dtype, device=device)]
    )
    dist.gather(padded_tensor, gather_tensors, dst=0)
    if rank == 0:
        gather_tensors = [
            tensor[:size.item()] for tensor, size in zip(gather_tensors, gather_shapes)
        ]
        return gather_tensors
    else:
        return None
