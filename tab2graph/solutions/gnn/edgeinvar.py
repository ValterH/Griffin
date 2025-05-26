from typing import Tuple, Dict, Optional, List, Any, Union

import pydantic
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from .hetero_invar import ORIGINAL_FEAT
from .Multihead_attention import MultiheadAttention


class EdgeINVARConvConfig(pydantic.BaseModel):
    has_bias: bool = False


class EdgeINVARBaseConv(nn.Module):

    def __init__(
        self,
        config: EdgeINVARConvConfig,
        in_size: Union[int, Tuple[int, int]],
        edge_in_size: int,
        out_size: int,
    ):
        super(EdgeINVARBaseConv, self).__init__()
        self.config = config

        self._in_size_src, self._in_size_dst = dgl.utils.expand_as_pair(in_size)
        self._edge_in_size = edge_in_size
        self._out_size = out_size

        self.fc_neigh = nn.Linear(self._in_size_src, out_size, bias=False)
        self.fc_self = nn.Linear(self._in_size_dst, out_size, bias=config.has_bias)

        if self._edge_in_size > 0:
            self.fc_edge = nn.Linear(self._edge_in_size, out_size, bias=False)

        self.mulihead_attention = MultiheadAttention(
            embed_dim=self._in_size_src,
            num_heads=8,
            dropout=0.1,
            self_attention=True,
        )

    def _compute_node_attention_aggre(self, graph, X_src):

        with graph.local_scope():
            graph.srcdata["h"] = self.fc_neigh(X_src)
            graph.update_all(
                dgl.function.u_mul_e("h", "e", "m"), dgl.function.mean("m", "neigh")
            )

            return graph.dstdata["neigh"]

    def _compute_node_edge_attention_aggre(self, graph, X_src, X_edge):

        with graph.local_scope():
            graph.srcdata["h"] = self.fc_neigh(X_src)
            graph.edata["e"] = self.fc_edge(X_edge) + graph.edata["e"]
            graph.update_all(
                dgl.function.u_mul_e("h", "e", "m"), dgl.function.mean("m", "neigh")
            )

            return graph.dstdata["neigh"]

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

    def _query_feat(
        self, original_feat, query_indices_row, query_indices_col, return_shape
    ):
        return_feat = torch.zeros(
            return_shape, dtype=original_feat.dtype, device=original_feat.device
        )
        return_feat[query_indices_row, query_indices_col] = original_feat
        return return_feat

    def forward(self, graph, X, X_edge=None, column_key_embedding_dict=None):
        # X: Tuple[Dict[str, torch.Tensor]]
        # check whether X satisfies the input size
        with graph.local_scope():
            # X_src, X_dst: Dict[str, torch.Tensor]
            X_src, X_dst = X

            # Handle the case of graphs without edges
            if graph.num_edges() == 0:
                if isinstance(X_dst[ORIGINAL_FEAT], torch.Tensor) and X_dst[
                    ORIGINAL_FEAT
                ].shape[0]:
                    return self.fc_self(X_dst[ORIGINAL_FEAT])
                else:
                    return torch.zeros(
                        (0, self._out_size), device=graph.edges()[0].device
                    )

            # Compute attention mask
            src_edges, dst_edges = graph.edges()
            original_src_feat = X_src[ORIGINAL_FEAT][src_edges]
            # attention_mask = dst_edges.view(-1, 1) == dst_edges.view(1, -1)
            # attention_weight_sum = torch.zeros_like(attention_mask, dtype=torch.float32)
            device = dst_edges.device
            unique_dst_edges, counts = torch.unique(dst_edges, return_counts=True)
            batch_size = unique_dst_edges.shape[0]
            max_count = counts.max()
            attention_weight_sum = torch.zeros(
                size=(batch_size, max_count, max_count),
                dtype=torch.float32,
                device=device,
            )
            attention_weight = torch.zeros(
                size=(batch_size, max_count, max_count),
                dtype=torch.float32,
                device=device,
            )
            padding_mask = (
                torch.arange(max_count, device=device).unsqueeze(0)
                < counts.unsqueeze(1)
            ).bool()
            attention_mask = padding_mask.unsqueeze(2) & padding_mask.unsqueeze(1)
            padding_mask = ~padding_mask
            range_tensor = torch.arange(max_count, dtype=torch.long, device=device)
            mask = range_tensor < counts.unsqueeze(1)
            indices_row_col = mask.nonzero(as_tuple=False)
            indices_row, indices_col = indices_row_col[:, 0], indices_row_col[:, 1]

            # Compute attention weight for each feature
            for feat_key, feat_value in X_src.items():
                if feat_key == ORIGINAL_FEAT:
                    continue
                # feat_value: torch.Tensor
                # feat_key: str
                # generate additional attention weight
                # for categorical features, the attention weight computed based on the equality of the feature values
                # for numerical features, the attention weight computed based on the difference of the feature values
                if _is_integer_tensor(feat_value):
                    feat = self._query_feat(
                        feat_value[src_edges],
                        indices_row,
                        indices_col,
                        (batch_size, max_count, feat_value.shape[-1]),
                    ).squeeze(-1)
                    # feat: batch_size * max_count
                    # attention_weight: batch_size * max_count * max_count
                    attention_weight = (
                        (feat.unsqueeze(2) == feat.unsqueeze(1)) & attention_mask
                    ).to(torch.float32)
                    # reversed_freq_feat_key = f"Reversed_Freq({feat_key})"
                    # reversed_freq_feat = X_src[reversed_freq_feat_key][src_edges].view(
                    #     -1
                    # )
                    # attention_weight[indices_row, indices_col, indices_col] = (
                    #     reversed_freq_feat
                    # )
                elif _is_float_tensor(feat_value):
                    pass
                    # feat = self._query_feat(
                    #     feat_value[src_edges],
                    #     indices_row,
                    #     indices_col,
                    #     (batch_size, max_count, feat_value.shape[-1]),
                    # ).squeeze(-1)
                    # attention_weight = (
                    #     torch.abs(feat.unsqueeze(2) - feat.unsqueeze(1))
                    #     * attention_mask
                    # )

                    # max_value_each_batch = torch.max(
                    #     torch.max(attention_weight, dim=2).values, dim=1
                    # ).values.view(-1, 1, 1)
                    # # normalize the attention weight to [0, 1] for each batch
                    # EPS = torch.tensor(1e-7, dtype=torch.float32, device=device)
                    # attention_weight = 1 - attention_weight / torch.maximum(
                    #     max_value_each_batch, EPS
                    # )
                    # attention_weight *= attention_mask
                    # reversed_freq_feat_key = (
                    #     f"Reversed_Freq({feat_key[10:-1]})"
                    #     if feat_key.startswith("TIMESTAMP")
                    #     else f"Reversed_Freq({feat_key})"
                    # )
                    # reversed_freq_feat = X_src[reversed_freq_feat_key][src_edges].view(
                    #     -1
                    # )
                    # attention_weight[indices_row, indices_col, indices_col] = (
                    #     reversed_freq_feat
                    # )
                else:
                    raise ValueError(
                        f"Unsupported feature type: feat_key={feat_key}, feat_value={feat_value}, dtype={feat_value.dtype}"
                    )

                # update the attention weight based on product of the feat_key and original features.
                # feat_key_embedding: llm_dim
                # original_feat: N * llm_dim
                # feat_key_relation_with_original_feat: N
                feat_key_embedding = column_key_embedding_dict[feat_key]
                src_feature_feat_key = original_src_feat @ feat_key_embedding
                # broadcast the feat_key_relation_with_original_feat to the attention_weight
                src_feature_feat_key_batch_wise = self._query_feat(
                    src_feature_feat_key,
                    indices_row,
                    indices_col,
                    (batch_size, max_count),
                )

                attention_weight = (
                    attention_weight * src_feature_feat_key_batch_wise.unsqueeze(1)
                )
                attention_weight_sum += attention_weight

            # Implment self-attention mechanism among the nodes
            query_feat = self._query_feat(
                original_src_feat,
                indices_row,
                indices_col,
                (batch_size, max_count, original_src_feat.shape[-1]),
            ).transpose(0, 1)
            attention_value, _ = self.mulihead_attention(
                query=query_feat,
                # key=query_feat,
                # value=query_feat,
                key_padding_mask=padding_mask,
                attn_bias=attention_weight_sum,
            )
            # edge_attention: max_count * batch_size * llm_dim
            # turn the edge_attention to the shape of the original feature
            edge_attention = attention_value[indices_row, indices_col]
            graph.edata["e"] = edge_attention

            if X_edge is None:
                h_neigh = self._compute_node_attention_aggre(
                    graph, X_src[ORIGINAL_FEAT]
                )
                # h_neigh = self._compute_node_src_aggre(graph, X_src[ORIGINAL_FEAT])
            else:
                h_neigh = self._compute_node_edge_attention_aggre(
                    graph, X_src[ORIGINAL_FEAT], X_edge
                )

            rst = self.fc_self(X_dst[ORIGINAL_FEAT]) + h_neigh

            return rst


class EdgeINVARRelationConv(nn.Module):
    def __init__(
        self,
        BaseConv: EdgeINVARBaseConv,
        relation_linear: nn.Linear,
        relation_embedding: torch.Tensor,
        column_key_linear: nn.Linear,
        column_key_embedding_dict: Dict[str, torch.Tensor],
    ):
        super(EdgeINVARRelationConv, self).__init__()
        self.BaseConv = BaseConv
        self.relation_linear = relation_linear
        self.relation_embedding = relation_embedding
        self.column_key_linear = column_key_linear
        self.column_key_embedding_dict = column_key_embedding_dict

    def forward(self, graph, X, X_edge=None):
        column_key_embedding_dict = {
            column_key: self.column_key_linear(column_key_embedding)
            for column_key, column_key_embedding in self.column_key_embedding_dict.items()
        }
        rst = self.BaseConv(graph, X, X_edge, column_key_embedding_dict)

        # rst: N * out_size
        # relation_embedding: hid_size
        # relation_linear: hid_size * out_size
        # should return hadamard product of relation_linear(relation_embedding) and rst

        if rst.shape[0]:
            return F.relu(rst + self.relation_linear(self.relation_embedding))
        else:
            return F.relu(rst)


def _is_integer_tensor(tensor: torch.Tensor) -> bool:
    return tensor.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]


def _is_float_tensor(tensor: torch.Tensor) -> bool:
    return tensor.dtype in [torch.float16, torch.float32, torch.float64]
