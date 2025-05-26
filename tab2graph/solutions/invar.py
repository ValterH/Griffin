import logging
from typing import Tuple, Dict, Optional, List, Any, Union
import pydantic
import copy

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import gml_solution
from .base_gml_solution import (
    BaseGNNSolutionConfig,
    BaseMultiTaskGMLSolution,
    BaseMultiTaskGNN,
)
from .graph_dataset_config import GraphConfig, GraphDatasetMultiTaskConfig
from .gnn import (
    EdgeINVARBaseConv,
    EdgeINVARRelationConv,
    EdgeINVARConvConfig,
    HeteroINVARLayer,
)


class INVARSolutionConfig(BaseGNNSolutionConfig):
    hid_size: int
    dropout: float
    conv: EdgeINVARConvConfig = EdgeINVARConvConfig()


class HeteroINVAR(nn.Module):
    def __init__(
        self,
        graph_config: GraphConfig,
        data_config: GraphDatasetMultiTaskConfig,
        solution_config: INVARSolutionConfig,
        node_in_size_dict: Dict[str, int],
        edge_in_size_dict: Dict[str, int],
        out_size: Optional[int],
        num_layers: int,
    ):
        super().__init__()
        if out_size is None:
            out_size = solution_config.hid_size
        self.out_size = out_size

        # assert num_layers > 0
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                cur_layer_in_size = node_in_size_dict
            else:
                cur_layer_in_size = {
                    ntype: solution_config.hid_size for ntype in graph_config.ntypes
                }
            if i == num_layers - 1:
                cur_layer_out_size = out_size
            else:
                cur_layer_out_size = solution_config.hid_size
            self.layers.append(
                HeteroINVARLayer(
                    graph_config,
                    data_config,
                    cur_layer_in_size,
                    cur_layer_out_size,
                    edge_in_size_dict,
                    EdgeINVARBaseConv,
                    EdgeINVARRelationConv,
                    solution_config.conv,
                )
            )
        self.dropout = nn.Dropout(solution_config.dropout)

    def forward(
        self,
        mfgs,
        X_node_dict: Dict[str, torch.Tensor],
        X_edge_dicts: List[Dict[str, torch.Tensor]],
        X_relation_dict: Dict[str, Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        if mfgs is None:
            mfgs = []
        assert len(mfgs) == len(self.layers)
        H_node_dict = X_node_dict
        for i, (layer, mfg, X_edge_dict) in enumerate(
            zip(self.layers, mfgs, X_edge_dicts)
        ):
            H_node_dict = layer(mfg, H_node_dict, X_edge_dict, X_relation_dict)
            if i != len(self.layers) - 1:
                H_node_dict = {
                    ntype: self.dropout(F.relu(H)) for ntype, H in H_node_dict.items()
                }
        return H_node_dict

    def copy_inner_embeddings_to_model_device(self, model_device):
        for layer in self.layers:
            layer.copy_inner_embeddings_to_model_device(model_device)


class INVAR(BaseMultiTaskGNN):

    def create_gnn(
        self,
        node_feat_size_dict: Dict[str, int],
        edge_feat_size_dict: Dict[str, int],
        out_size: Optional[int],
    ) -> nn.Module:
        gnn = HeteroINVAR(
            self.data_config.graph,
            self.data_config,
            self.solution_config,
            node_feat_size_dict,
            edge_feat_size_dict,
            out_size,
            num_layers=len(self.solution_config.fanouts),
        )
        return gnn


@gml_solution
class INVARSolution(BaseMultiTaskGMLSolution):

    config_class = INVARSolutionConfig
    name = "invar"

    def create_model(self):
        return INVAR(self.solution_config, self.data_config)
