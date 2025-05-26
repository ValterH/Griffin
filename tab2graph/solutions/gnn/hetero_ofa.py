from typing import Tuple, Dict, Optional, List, Any, Union

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..graph_dataset_config import GraphConfig

from sentence_transformers import SentenceTransformer

LLM_DIM_DICT = {
    "ST": 768,
    "nomic": 256,
}

class HeteroGNNOFALayer(nn.Module):
    """Heterogeneous GNN layer wrapper."""

    def __init__(
        self,
        graph_config: GraphConfig,
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
        assert len(size_config_dict) == 1, "Only one size configuration should be present."
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
        assert len(node_size_config_dict) == 1, "Only one size configuration should be present."

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

        self.conv = dgl.nn.HeteroGraphConv(
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
                        f"Edge type: from {st} to {dt} by {et}",
                        batch_size=1,
                        convert_to_tensor=True,
                        convert_to_numpy=False,
                        prompt="classification: " if LLM_name == "nomic" else None,
                    ),
                )
                for (st, et, dt) in etypes
            },
            aggregate="sum",
        )
        # self.loop_fc = nn.ModuleDict(
        #     {
        #         ntype: hetero_self_loop_config_dict[
        #             node_size_config_dict[in_size_dict[ntype]]
        #         ]
        #         for ntype in graph_config.ntypes
        #     }
        # )
        self.loop_ntype_embedding = {
            ntype: self.relation_embedding_model.encode(
                f"Node type: {ntype}",
                batch_size=1,
                convert_to_tensor=True,
                convert_to_numpy=False,
                prompt="classfication: " if LLM_name == "nomic" else None,
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
    ) -> Dict[str, torch.Tensor]:
        # Convert edge dict key from src_type:etype:dst_type to etype.
        X_edge_dict = {
            key.split(":")[1]: {"X_edge": X} for key, X in X_edge_dict.items()
        }
        H_node_dict = self.conv(mfg, X_node_dict, mod_kwargs=X_edge_dict)
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
        }
        return H_node_dict
