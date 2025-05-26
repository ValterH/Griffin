from enum import Enum
from typing import Tuple, Dict, Optional, List, Any
import torch
import torch.nn as nn
import pydantic
from dbinfer_bench import DBBTaskType

from .graph_dataset_config import TaskConfig

class PredictorConfig(pydantic.BaseModel):
    num_layers : Optional[int] = 1
    hid_size : Optional[int] = 128
    dropout : Optional[float] = 0.3
    gradual_decrease_hid_size : Optional[bool] = False
    Griffin_predictor : Optional[bool] = False
    # with_seed_ctx_features means whether the data configs including seed context features.
    with_seed_ctx_features: Optional[bool] = True

class Predictor(nn.Module):
    def __init__(
        self,
        task_config : TaskConfig,
        predictor_config : PredictorConfig,
        seed_embed_size : int,
        seed_ctx_embed_size : int
    ):
        super().__init__()
        self.task_config = task_config
        self.predictor_config = predictor_config

        if predictor_config.Griffin_predictor:
            # Use Griffin's prediction.
            if task_config.task_type == DBBTaskType.retrieval:
                assert (
                    task_config.num_seeds == 2
                ), "Griffin's prediction unifies all tasks to edge-level prediction."
                if predictor_config.with_seed_ctx_features:
                    assert (
                        seed_ctx_embed_size == seed_embed_size
                    ), "Griffin's prediction requires seed_ctx_embed_size == seed_embed_size."
                    in_size = seed_embed_size * 4
                else:
                    in_size = seed_embed_size * 2
            elif task_config.task_type == DBBTaskType.regression:
                assert task_config.num_seeds == 1, "Regression task should have only one seed."
                if predictor_config.with_seed_ctx_features:
                    in_size = seed_embed_size * 2
                else:
                    in_size = seed_embed_size
        else:
            in_size = seed_embed_size * task_config.num_seeds + seed_ctx_embed_size
        out_size = self.get_out_size(task_config)

        if predictor_config.num_layers == 1:
            # Use one linear layer.
            self.model = nn.Linear(in_size, out_size)

        else:
            # Use an MLP.
            self.model = nn.Sequential(
                nn.Linear(in_size, predictor_config.hid_size),
                nn.ReLU(),
                nn.Dropout(self.predictor_config.dropout)
            )
            hid_size_in = predictor_config.hid_size
            if predictor_config.gradual_decrease_hid_size:
                for i in range(predictor_config.num_layers - 2):
                    self.model += nn.Sequential(
                        nn.Linear(hid_size_in, int(hid_size_in / 2)),
                        nn.ReLU(),
                        nn.Dropout(self.predictor_config.dropout)
                    )
                    hid_size_in = int(hid_size_in / 2)
                self.model.append(nn.Linear(hid_size_in, out_size))
            else:
                for i in range(predictor_config.num_layers - 1):
                    self.model += nn.Sequential(
                        nn.Linear(predictor_config.hid_size, predictor_config.hid_size),
                        nn.ReLU(),
                        nn.Dropout(self.predictor_config.dropout)
                    )
                self.model.append(nn.Linear(predictor_config.hid_size, out_size))

    def forward(self, seed_embeds, seed_ctx_embeds):
        """Forward

        Input shape
          seed_embeds : (N, K1, D1) or (N, D1)
          (optional) seed_ctx_embeds : (N, D2)

        Output logits or target of shape
          ret: (N, C). If C == 1, shape (N, )
        """
        if self.predictor_config.Griffin_predictor:
            if self.predictor_config.with_seed_ctx_features:
                # Edge-level prediction.
                if seed_embeds.dim() == 3:
                    src_embeds = seed_embeds[:, 0, :]
                    dst_embeds = seed_embeds[:, 1, :]
                    embeds = torch.cat(
                        [
                            src_embeds + dst_embeds,
                            src_embeds * dst_embeds,
                            seed_ctx_embeds + dst_embeds,
                            seed_ctx_embeds * dst_embeds,
                        ],
                        dim=1,
                    )
                else:
                    embeds = torch.cat([seed_embeds, seed_ctx_embeds], dim=1)
            else:
                if seed_embeds.dim() == 3:
                    # embeds = torch.cat([seed_embeds[:, 0, :], seed_embeds[:, 1, :]], dim=1)
                    embeds = torch.cat(
                        [
                            seed_embeds[:, 0, :] + seed_embeds[:, 1, :],
                            seed_embeds[:, 0, :] * seed_embeds[:, 1, :],
                        ],
                        dim=1,
                    )
                else:
                    embeds = seed_embeds
        else:
            N = seed_embeds.shape[0]
            embeds = seed_embeds.view(N, -1)
            if seed_ctx_embeds is not None:
                embeds = torch.cat([embeds, seed_ctx_embeds], dim=1)
        return self.model(embeds).squeeze(-1)

    @staticmethod
    def get_out_size(task_config : TaskConfig) -> int:
        if task_config.task_type == DBBTaskType.classification:
            out_size = task_config.num_classes
        elif task_config.task_type == DBBTaskType.regression:
            out_size = 1
        elif task_config.task_type == DBBTaskType.retrieval:
            out_size = 1
        else:
            raise ValueError(f"Unsupported task type {task_config.task_type}.")
        return out_size

class SeedLookup(nn.Module):
    def __init__(self, target_type):
        super().__init__()
        self.target_type = target_type

    def forward(
        self,
        node_embed_dict : Dict[str, torch.Tensor],
        seed_lookup_idx : torch.Tensor,
    ) -> torch.Tensor:
        """Look up seed embeddings from node embeddings.

        Input shape:
            node_embed_dict : each item is of shape (M_t, D)
            seed_lookup_idx : (N, K) or None
                None means identity.

        Output shape:
            seed_embeds : (N, K, D)
        """
        if len(self.target_type.split(":")) != 3:
            # Node-level prediction.
            node_embed = node_embed_dict[self.target_type]
            if seed_lookup_idx is None:
                return node_embed
            else:
                return node_embed[seed_lookup_idx]
        else:
            # Edge-level prediction.
            src_type, _, dst_type = self.target_type.split(":")
            src_embed = node_embed_dict[src_type]
            dst_embed = node_embed_dict[dst_type]
            if seed_lookup_idx is None:
                src_seed_embed = src_embed
                dst_seed_embed = dst_embed
            else:
                src_seed_embed = src_embed[seed_lookup_idx[:,0]]
                dst_seed_embed = dst_embed[seed_lookup_idx[:,1]]
            return torch.stack([src_seed_embed, dst_seed_embed], dim=1)
