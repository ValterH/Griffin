from typing import Tuple, Dict, Optional, List, Any, Union

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..graph_dataset_config import GraphConfig, GraphDatasetMultiTaskConfig

from sentence_transformers import SentenceTransformer

LLM_DIM_DICT = {
    "ST": 768,
    "nomic": 256,
}
ORIGINAL_FEAT = "full_node_feature"


# Implement the HeteroGraphConv_NestedDictInput class.
# The class inherits from dgl.nn.HeteroGraphConv.
# The only updated method is the forward method, allowing for nested dictionary inputs.
class HeteroGraphConv_NestedDictInput(dgl.nn.HeteroGraphConv):
    def forward(self, g, inputs, mod_args=None, mod_kwargs=None):
        """Forward computation

        Invoke the forward function with each module and aggregate their results.

        Parameters
        ----------
        g : DGLGraph
            Graph data.
        ! Note: inputs can be a nested dictionary.
        inputs : dict[str, dict[str, Tensor]]
            Input node features.
        mod_args : dict[str, tuple[any]], optional
            Extra positional arguments for the sub-modules.
        mod_kwargs : dict[str, dict[str, any]], optional
            Extra key-word arguments for the sub-modules.

        Returns
        -------
        dict[str, Tensor]
            Output representations for every types of nodes.
        """
        if mod_args is None:
            mod_args = {}
        if mod_kwargs is None:
            mod_kwargs = {}
        outputs = {nty: [] for nty in g.dsttypes}
        # We only consider the case: g is a block, inputs is a nested dictionary.
        assert (
            isinstance(inputs, dict) and g.is_block
        ), "The input should be a nested dictionary and g should be a block."
        src_inputs = inputs
        dst_inputs = {
            k: {k2: v2[: g.number_of_dst_nodes(k)] for k2, v2 in v.items()}
            for k, v in inputs.items()
        }

        for stype, etype, dtype in g.canonical_etypes:
            rel_graph = g[stype, etype, dtype]
            if stype not in src_inputs or dtype not in dst_inputs:
                continue
            dstdata = self._get_module((stype, etype, dtype))(
                rel_graph,
                (src_inputs[stype], dst_inputs[dtype]),
                *mod_args.get(etype, ()),
                **mod_kwargs.get(etype, {}),
            )
            outputs[dtype].append(dstdata)
        rsts = {}
        for nty, alist in outputs.items():
            if len(alist) != 0:
                rsts[nty] = self.agg_fn(alist, nty)
        return rsts


class HeteroINVARLayer(nn.Module):
    """Heterogeneous GNN layer wrapper."""

    def __init__(
        self,
        graph_config: GraphConfig,
        data_config: GraphDatasetMultiTaskConfig,
        in_size_dict: Dict[str, int],
        out_size: int,
        edge_in_size_dict: Dict[str, int],
        layer_class_base,
        layer_class_relation,
        layer_config,
        LLM_name: str = "nomic",
    ):
        super().__init__()
        etypes = [tuple(et.split(":")) for et in graph_config.etypes]
        # Create a dictionary of layer configurations for each unique size configuration.
        hetero_conv_config_dict = {}
        size_config_dict = {}
        for st, et, dt in etypes:
            if (
                in_size_dict[st],
                in_size_dict[dt],
                edge_in_size_dict[f"{st}:{et}:{dt}"],
            ) not in size_config_dict:
                size_config_dict[
                    (
                        in_size_dict[st],
                        in_size_dict[dt],
                        edge_in_size_dict[f"{st}:{et}:{dt}"],
                    )
                ] = (st, et, dt)
        # TODO: The current implementation constructs a unique layer for each unique size configuration.
        # However, we should identify that by considering edge types.
        # To do so, we add a simple assert on the number of unique size configurations.
        assert (
            len(size_config_dict) == 1
        ), "Only one size configuration should be present."
        for size_config, (st, et, dt) in size_config_dict.items():
            hetero_conv_config_dict[(st, et, dt)] = layer_class_base(
                layer_config,
                (in_size_dict[st], in_size_dict[dt]),
                size_config[2],
                out_size,
            )

        # Create a dictionary of node self loop configurations for each unique size configuration.
        # hetero_self_loop_config_dict = {}
        node_size_config_dict = {}
        for ntype in graph_config.ntypes:
            if in_size_dict[ntype] not in size_config_dict:
                node_size_config_dict[in_size_dict[ntype]] = ntype
        # for size_config, ntype in node_size_config_dict.items():
        #     hetero_self_loop_config_dict[ntype] = nn.Linear(
        #         in_size_dict[ntype], LLM_DIM_DICT[LLM_name]
        #     )
        assert (
            len(node_size_config_dict) == 1
        ), "Only one size configuration should be present."

        if LLM_name == "ST":
            self.relation_embedding_model = SentenceTransformer(
                "multi-qa-distilbert-cos-v1",
                device="cuda:0",
                cache_folder="cache_data/model",
            )
        elif LLM_name == "nomic":
            self.relation_embedding_model = SentenceTransformer(
                "nomic-ai/nomic-embed-text-v1.5",
                device="cuda:0",
                cache_folder="cache_data/model",
                trust_remote_code=True,
                truncate_dim=LLM_DIM_DICT[LLM_name],
            )
        else:
            raise ValueError(f"Unknown LLM name: {LLM_name}")

        self.relation_linear = nn.Linear(LLM_DIM_DICT[LLM_name], out_size)

        self.column_key_embedding_dict = {
            node_type: {
                feat_key: self.relation_embedding_model.encode(
                    f"Column name {feat_key} of node type {node_type}",
                    batch_size=1,
                    convert_to_tensor=True,
                    convert_to_numpy=False,
                    prompt="clustering: " if LLM_name == "nomic" else None,
                )
                for feat_key in node_feat.keys()
            }
            for node_type, node_feat in data_config.node_features.items()
        }
        self.column_key_linear = nn.Linear(LLM_DIM_DICT[LLM_name], out_size)

        self.conv = HeteroGraphConv_NestedDictInput(
            {
                (st, et, dt): layer_class_relation(
                    BaseConv=hetero_conv_config_dict[
                        size_config_dict[
                            (
                                in_size_dict[st],
                                in_size_dict[dt],
                                edge_in_size_dict[f"{st}:{et}:{dt}"],
                            )
                        ]
                    ],
                    relation_linear=self.relation_linear,
                    relation_embedding=self.relation_embedding_model.encode(
                        f"Edge type: from {st} to {dt} by {et.split('-')[-1]}",
                        batch_size=1,
                        convert_to_tensor=True,
                        convert_to_numpy=False,
                        prompt="clustering: " if LLM_name == "nomic" else None,
                    ),
                    column_key_linear=self.column_key_linear,
                    column_key_embedding_dict=self.column_key_embedding_dict[st],
                )
                for (st, et, dt) in etypes
            },
            aggregate="sum",
        )
        self.loop_ntype_embedding = {
            ntype: self.relation_embedding_model.encode(
                f"Node type: {ntype}",
                batch_size=1,
                convert_to_tensor=True,
                convert_to_numpy=False,
                prompt="clustering: " if LLM_name == "nomic" else None,
            )
            for ntype in graph_config.ntypes
        }

        self.ntype_linear = nn.Linear(LLM_DIM_DICT[LLM_name], out_size, bias=False)
        self.final_loop_fc = nn.Linear(out_size, out_size, bias=False)

        # del embedding model
        del self.relation_embedding_model

    def forward(
        self,
        mfg,
        X_node_dict: Dict[str, torch.Tensor],
        X_edge_dict: Dict[str, torch.Tensor],
        X_relation_dict: Dict[str, Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        # Convert edge dict key from src_type:etype:dst_type to etype.
        X_edge_dict = {
            key.split(":")[1]: {"X_edge": X} for key, X in X_edge_dict.items()
        }
        # Combine X_node_dict and X_relation_dict
        # X_node_dict_combine : Dict[str, Dict[str, torch.Tensor]]
        # For each node type, the X_node_dict_combine contains:
        # ORIGINAL_FEAT: X_node_dict[ntype]
        # feat_type: X_relation_dict[ntype][feat_type]
        X_node_dict_combine = {}
        for ntype in X_node_dict:
            X_node_dict_combine[ntype] = {
                ORIGINAL_FEAT: X_node_dict[ntype],
            }
            X_node_dict_combine[ntype].update(X_relation_dict[ntype])
        H_node_dict = self.conv(mfg, X_node_dict_combine, mod_kwargs=X_edge_dict)
        X_dstnode_dict = {
            ntype: X[: mfg.number_of_dst_nodes(ntype)]
            for ntype, X in X_node_dict.items()
        }
        H_node_dict = {
            # ntype: H_node_dict[ntype] + self.loop_fc[ntype](X_dstnode_dict[ntype])
            ntype: H_node_dict[ntype]
            + self.final_loop_fc(
                X_dstnode_dict[ntype]
                + self.ntype_linear(self.loop_ntype_embedding[ntype])
                # * self.loop_fc[ntype](X_dstnode_dict[ntype])
            )
            for ntype in H_node_dict
            if H_node_dict[ntype].shape[0]
        }
        return H_node_dict

    def copy_inner_embeddings_to_model_device(self, model_device):
        # Move the column key embeddings to the model device
        for ntype, feat_dict in self.column_key_embedding_dict.items():
            for feat_key, embedding in feat_dict.items():
                self.column_key_embedding_dict[ntype][feat_key] = embedding.to(model_device)

        # Move the relation embedding of conv to the model device
        for etype, conv in self.conv.mods.items():
            conv.relation_embedding = conv.relation_embedding.to(model_device)

        # Move the loop ntype embedding to the model device
        self.loop_ntype_embedding = {
            ntype: embedding.to(model_device)
            for ntype, embedding in self.loop_ntype_embedding.items()
        }
