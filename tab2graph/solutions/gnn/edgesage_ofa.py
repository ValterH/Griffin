from typing import Tuple, Dict, Optional, List, Any, Union

import pydantic
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeSAGEOFAConvConfig(pydantic.BaseModel):
    has_bias: bool = False


class EdgeSAGEOFABaseConv(nn.Module):

    def __init__(
        self,
        config: EdgeSAGEOFAConvConfig,
        in_size: Union[int, Tuple[int, int]],
        edge_in_size: int,
        out_size: int,
    ):
        super(EdgeSAGEOFABaseConv, self).__init__()
        self.config = config

        self._in_size_src, self._in_size_dst = dgl.utils.expand_as_pair(in_size)
        self._edge_in_size = edge_in_size
        self._out_size = out_size

        self.fc_neigh = nn.Linear(self._in_size_src, out_size, bias=False)
        self.fc_self = nn.Linear(self._in_size_dst, out_size, bias=config.has_bias)

        if self._edge_in_size > 0:
            self.fc_edge = nn.Linear(self._edge_in_size, out_size, bias=False)

    def _compute_node_src_aggre(self, graph, X_src):

        with graph.local_scope():

            graph.srcdata["h"] = self.fc_neigh(X_src)
            graph.update_all(
                dgl.function.copy_u("h", "m"), dgl.function.mean("m", "neigh")
            )

            return graph.dstdata["neigh"]

    def _compute_edge_aggre(self, graph, X_edge):

        with graph.local_scope():

            assert X_edge.shape[0] == graph.num_edges()

            graph.edata["e"] = self.fc_edge(X_edge)
            graph.update_all(
                dgl.function.copy_e("e", "m_e"), dgl.function.mean("m_e", "neigh_e")
            )

            return graph.dstdata["neigh_e"]

    def _compute_node_edge_aggre(self, graph, X_src, X_edge):

        with graph.local_scope():

            assert X_edge.shape[0] == graph.num_edges()

            graph.srcdata["h"] = self.fc_neigh(X_src)
            graph.edata["e"] = self.fc_edge(X_edge)
            graph.update_all(
                dgl.function.u_mul_e("h", "e", "m"), dgl.function.mean("m", "neigh")
            )

            return graph.dstdata["neigh"]

    def forward(self, graph, X, X_edge=None):
        with graph.local_scope():
            if isinstance(X, tuple):
                X_src, X_dst = X
            else:
                X_src = X_dst = X
                if graph.is_block:
                    X_dst = X_src[: graph.number_of_dst_nodes()]

            # Handle the case of graphs without edges
            if graph.num_edges() == 0:
                return self.fc_self(X_dst)
            # Message Passing
            if X_edge is None:
                h_neigh = self._compute_node_src_aggre(graph, X_src)
            else:
                h_neigh = self._compute_node_edge_aggre(graph, X_src, X_edge)

            rst = self.fc_self(X_dst) + h_neigh

            return rst


class EdgeSAGEOFARelationConv(nn.Module):
    def __init__(
        self,
        BaseConv: EdgeSAGEOFABaseConv,
        relation_linear: nn.Linear,
        relation_embedding: torch.Tensor,
    ):
        super(EdgeSAGEOFARelationConv, self).__init__()
        self.BaseConv = BaseConv
        self.relation_linear = relation_linear
        self.relation_embedding = relation_embedding

    def forward(self, graph, X, X_edge=None):
        rst = self.BaseConv(graph, X, X_edge)

        # rst: N * out_size
        # relation_embedding: hid_size
        # relation_linear: hid_size * out_size
        # should return hadamard product of relation_linear(relation_embedding) and rst

        if rst.shape[0]:
            return F.relu(rst + self.relation_linear(self.relation_embedding))
        else:
            return F.relu(rst)
