"""Composite class to encode a dictionary of features based on their config."""

from collections import defaultdict
import logging
from typing import Tuple, Dict, Optional, List, Any
import torch
import torch.nn as nn

from .base import get_encoder_class
from .id import IdentityEncoder
from ..graph_dataset_config import (
    GraphDatasetConfig,
    GraphDatasetMultiTaskConfig,
    FeatureConfig,
)
from dbinfer_bench import DBBColumnDType
from .numeric import OrthogonalEncoder

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")

LLM_DIM_DICT = {
    "ST": 768,
    "nomic": 256,
}


class SelfAttentionAggregator(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        num_heads=4,
        num_layer=1,
        dim_feedforward: int = None,
        dropout=0.1,
        LLM_name: str = "nomic",
    ):
        super().__init__()
        assert in_size == out_size
        if dim_feedforward is None:
            dim_feedforward = 4 * in_size
        self.attention_layer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=in_size,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                norm_first=True,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=num_layer,
        )
        self.linq = nn.Linear(
            LLM_DIM_DICT[LLM_name], in_size, bias=False
        )  # nn.Sequential(nn.Linear(in_size, out_size, bias=False), nn.SiLU(inplace=True))
        self.crossattention = nn.MultiheadAttention(
            in_size, num_heads, dropout, bias=False, batch_first=True
        )
        self.column_name_fuse = nn.Sequential(
            nn.Linear(in_size + LLM_DIM_DICT[LLM_name], in_size),
            nn.SiLU(inplace=True),
            nn.Linear(in_size, in_size, bias=False),
        )

    def forward(self, tar, column_name_emb, x, mask=None):
        # tar: (batch_size, q_len, in_size)
        # x: (batch_size, seq_len, in_size)
        # column_name_emb: (seq_len, in_size)
        # fuse column name embedding with x
        x = self.column_name_fuse(torch.cat([x, column_name_emb], dim=-1))
        x = self.attention_layer(x, src_key_padding_mask=mask)
        q = self.linq(tar)
        ret = self.crossattention(q, x, x, key_padding_mask=mask, need_weights=False)[0]
        return ret.squeeze(1)


class FeatDictEncoder(nn.Module):

    def __init__(
        self,
        feat_configs: Dict[Any, FeatureConfig],
        feature_groups: Optional[List[List[Any]]],
        feat_encode_size: Optional[int],
    ):
        super().__init__()
        fg_cfgs = []
        self.ft2gid = {}
        if feature_groups is not None:
            for fg in feature_groups:
                gid = len(fg_cfgs)
                fg_cfgs.append(feat_configs[fg[0]])
                for ft in fg:
                    self.ft2gid[ft] = gid
        for ft in sorted(feat_configs.keys()):
            cfg = feat_configs[ft]
            if ft not in self.ft2gid:
                self.ft2gid[ft] = len(fg_cfgs)
                fg_cfgs.append(cfg)

        # Create encoders.
        self.encoders = nn.ModuleList()
        for i, cfg in enumerate(fg_cfgs):
            encoder_class = get_encoder_class(cfg.dtype)
            self.encoders.append(encoder_class(cfg.extra_fields, feat_encode_size))

        self.out_size_dict = {
            ft: self.encoders[gid].out_size for ft, gid in self.ft2gid.items()
        }

    def forward(
        self, input_feat_dict: Dict[Any, torch.Tensor]
    ) -> Dict[Any, torch.Tensor]:
        return {
            ft: self.encoders[self.ft2gid[ft]](val)
            for ft, val in input_feat_dict.items()
            # FIXME: what to do if a key in input_feat_dict does not exist in the encoders?
            # (e.g. itemId in Diginetica-clicks shouldn't be encoded?)
            if ft in self.ft2gid
        }

    def __repr__(self):
        super_repr = super().__repr__()
        ft_groups = [[] for i in range(len(self.encoders))]
        for ft, gid in self.ft2gid.items():
            ft_groups[gid].append(ft)
        ft_group_str = [str(fg) for fg in ft_groups]
        extra_repr = "  feat_groups=[\n"
        for fg in ft_groups:
            extra_repr += f"    {fg}\n"
        extra_repr += "  ]\n)"
        return super_repr[:-1] + extra_repr


FeatDict = Dict[str, Dict[str, torch.Tensor]]


class GraphFeatDictEncoder(nn.Module):

    @staticmethod
    def is_valid_feat(cfg: FeatureConfig) -> bool:
        return (
            not cfg.is_time
            and get_encoder_class(cfg.dtype, allow_missing=True) is not None
        )

    def __init__(
        self, data_config: GraphDatasetConfig, feat_encode_size: Optional[int]
    ):
        super().__init__()
        # Get feature groups.
        feature_groups = []
        if data_config.feature_groups is not None:
            feature_groups += [
                [(ft.type, ft.name) for ft in fg] for fg in data_config.feature_groups
            ]

        def _get_or_add_fg(feat_type, feat_name):
            for fg in feature_groups:
                for ty, name in fg:
                    if ty == feat_type and name == feat_name:
                        return fg
            feature_groups.append([(feat_type, feat_name)])
            return feature_groups[-1]

        # Get feature configs.
        feat_configs = {}
        for ntype, nt_cfgs in data_config.node_features.items():
            for feat_name, cfg in nt_cfgs.items():
                if self.is_valid_feat(cfg):
                    feat_configs[(ntype, feat_name)] = cfg
                else:
                    _raise_ignore_warning(ntype, feat_name, cfg)
        for etype, et_cfgs in data_config.edge_features.items():
            for feat_name, cfg in et_cfgs.items():
                if self.is_valid_feat(cfg):
                    feat_configs[(etype, feat_name)] = cfg
                else:
                    _raise_ignore_warning(etype, feat_name, cfg)
        target_type = data_config.task.target_type
        for feat_name in sorted(data_config.seed_features.keys()):
            cfg = data_config.seed_features[feat_name]
            if self.is_valid_feat(cfg):
                feat_configs[("__seed__", feat_name)] = cfg
                if (target_type, feat_name) in feat_configs:
                    _get_or_add_fg(target_type, feat_name).append(
                        ("__seed__", feat_name)
                    )
            else:
                _raise_ignore_warning("__seed__", feat_name, cfg)

        self.encoder = FeatDictEncoder(feat_configs, feature_groups, feat_encode_size)

        self.node_out_size_dict = defaultdict(lambda: 0)
        for ntype, nt_cfgs in data_config.node_features.items():
            for feat_name in nt_cfgs:
                self.node_out_size_dict[ntype] += self.encoder.out_size_dict.get(
                    (ntype, feat_name), 0
                )

        self.edge_out_size_dict = defaultdict(lambda: 0)
        for etype, et_cfgs in data_config.edge_features.items():
            for feat_name in et_cfgs:
                self.edge_out_size_dict[etype] += self.encoder.out_size_dict.get(
                    (etype, feat_name), 0
                )

        self.seed_ctx_out_size = 0
        for feat_name in data_config.seed_features:
            self.seed_ctx_out_size += self.encoder.out_size_dict.get(
                ("__seed__", feat_name), 0
            )

    def forward(
        self,
        feat_dict: FeatDict,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Forward function.

        Support both node feature dict and edge feature dict. For edge feature dict,
        the edge type is a string of form "src_type:edge_type:dst_type".
        """
        flat_feat_dict = {}
        for ty, ty_feat_dict in feat_dict.items():
            for feat_name, feat in ty_feat_dict.items():
                flat_feat_dict[(ty, feat_name)] = feat

        flat_feat_dict = self.encoder(flat_feat_dict)

        # Revert to nested dict.
        new_feat_dict = {}
        for ty, ty_feat_dict in feat_dict.items():
            if ty not in new_feat_dict:
                new_feat_dict[ty] = {}
            for feat_name, feat in ty_feat_dict.items():
                new_feat_dict[ty][feat_name] = flat_feat_dict[(ty, feat_name)]
        return new_feat_dict

    def __repr__(self):
        super_repr = super().__repr__()
        extra_repr = (
            f"  node_out_size_dict={dict(self.node_out_size_dict)}\n"
            f"  edge_out_size_dict={dict(self.edge_out_size_dict)}\n"
            f"  seed_ctx_out_size={self.seed_ctx_out_size}\n)"
        )
        return super_repr[:-1] + extra_repr


class FeatDictEncoderMultiTask(nn.Module):

    def __init__(
        self,
        feat_configs: Dict[Any, FeatureConfig],
        feature_groups: Optional[List[List[Any]]],
        feat_encode_size: Optional[int],
        use_one_encoder_for_llm: Optional[bool] = False,
        with_original_features: Optional[bool] = False,
        llm_dim: Optional[int] = 256,
    ):
        super().__init__()
        fg_cfgs = []
        self.ft2gid = {}
        if feature_groups is not None:
            for fg in feature_groups:
                gid = len(fg_cfgs)
                fg_cfgs.append(feat_configs[fg[0]])
                for ft in fg:
                    self.ft2gid[ft] = gid
        for ft in sorted(feat_configs.keys()):
            cfg = feat_configs[ft]
            if ft not in self.ft2gid:
                self.ft2gid[ft] = len(fg_cfgs)
                fg_cfgs.append(cfg)

        # Create encoders.
        self.encoders = nn.ModuleList()
        if use_one_encoder_for_llm:
            # First check if all feature groups have the same dtype and size
            if not with_original_features:
                # If with_original_features is False, then all feature groups must have the same dtype
                for i, cfg in enumerate(fg_cfgs):
                    if cfg.dtype != fg_cfgs[0].dtype:
                        raise ValueError(
                            "All feature groups must have the same dtype if use_one_encoder_for_llm=True."
                        )
                    if (
                        "in_size" in cfg.extra_fields
                        and cfg.extra_fields["in_size"]
                        != fg_cfgs[0].extra_fields["in_size"]
                    ):
                        raise ValueError(
                            "All feature groups must have the same in_size if use_one_encoder_for_llm=True."
                        )
                encoder_class = get_encoder_class(fg_cfgs[0].dtype)
                self.encoders.append(
                    encoder_class(fg_cfgs[0].extra_fields, feat_encode_size)
                )
            else:
                # If with_original_features is True, then we only keep features with dim as llm_dim
                encoder_class = get_encoder_class(DBBColumnDType.float_t)
                self.encoders.append(
                    encoder_class({"in_size": llm_dim}, feat_encode_size)
                )
                # For all other feature groups, we assign them to zero, i.e., an encoder that has no input, but always outputs 0
                self.encoders.append(OrthogonalEncoder(out_size=feat_encode_size))
                self.encoders.append(IdentityEncoder(out_size=feat_encode_size))
        else:
            for i, cfg in enumerate(fg_cfgs):
                encoder_class = get_encoder_class(cfg.dtype)
                self.encoders.append(encoder_class(cfg.extra_fields, feat_encode_size))
        # Update all values in ft2gid to 0 if use_one_encoder_for_llm is True
        if use_one_encoder_for_llm:
            if not with_original_features:
                self.ft2gid = {ft: 0 for ft in self.ft2gid}
            else:
                # for features with dim as llm_dim, we assign them to 0
                # else, we assign them to 1
                for ft in self.ft2gid:
                    # For Griffin text feature, we use numeric encoder from llm_dim to feat_encode_size
                    if ft[1].startswith("Griffin_text_"):
                        self.ft2gid[ft] = 0
                    # for numerical features, we use orthogonal encoder
                    elif feat_configs[ft].dtype == DBBColumnDType.float_t:
                        self.ft2gid[ft] = 1
                    else:
                        self.ft2gid[ft] = 2

        self.out_size_dict = {
            ft: self.encoders[gid].out_size for ft, gid in self.ft2gid.items()
        }

    def forward(
        self, input_feat_dict: Dict[Any, torch.Tensor]
    ) -> Dict[Any, torch.Tensor]:
        return {
            ft: self.encoders[self.ft2gid[ft]](val)
            for ft, val in input_feat_dict.items()
            # FIXME: what to do if a key in input_feat_dict does not exist in the encoders?
            # (e.g. itemId in Diginetica-clicks shouldn't be encoded?)
            if ft in self.ft2gid
        }

    def __repr__(self):
        super_repr = super().__repr__()
        ft_groups = [[] for i in range(len(self.encoders))]
        for ft, gid in self.ft2gid.items():
            ft_groups[gid].append(ft)
        ft_group_str = [str(fg) for fg in ft_groups]
        extra_repr = "  feat_groups=[\n"
        for fg in ft_groups:
            extra_repr += f"    {fg}\n"
        extra_repr += "  ]\n)"
        return super_repr[:-1] + extra_repr


class GraphFeatDictMultiTaskEncoder(nn.Module):

    @staticmethod
    def is_valid_feat(cfg: FeatureConfig) -> bool:
        return (
            not cfg.is_time
            and get_encoder_class(cfg.dtype, allow_missing=True) is not None
        )

    def __init__(
        self,
        data_config: GraphDatasetMultiTaskConfig,
        feat_encode_size: Optional[int],
        use_one_encoder_for_llm: bool = False,
        # whether different columns of the same table should be added or concatenated
        add_columns_dim_together: bool = True,
        with_original_features: bool = False,
        llm_dim: int = 256,
    ):
        super().__init__()
        # Get feature groups.
        feature_groups = []
        if data_config.feature_groups is not None:
            feature_groups += [
                [(ft.type, ft.name) for ft in fg] for fg in data_config.feature_groups
            ]

        def _get_or_add_fg(feat_type, feat_name):
            for fg in feature_groups:
                for ty, name in fg:
                    if ty == feat_type and name == feat_name:
                        return fg
            feature_groups.append([(feat_type, feat_name)])
            return feature_groups[-1]

        # Get feature configs.
        feat_configs = {}
        for ntype, nt_cfgs in data_config.node_features.items():
            for feat_name, cfg in nt_cfgs.items():
                if self.is_valid_feat(cfg):
                    feat_configs[(ntype, feat_name)] = cfg
                else:
                    _raise_ignore_warning(ntype, feat_name, cfg)
        for etype, et_cfgs in data_config.edge_features.items():
            for feat_name, cfg in et_cfgs.items():
                if self.is_valid_feat(cfg):
                    feat_configs[(etype, feat_name)] = cfg
                else:
                    _raise_ignore_warning(etype, feat_name, cfg)
        for task_name, task in data_config.task_dict.items():
            target_type = task.target_type
            for feat_name in sorted(data_config.seed_features_dict[task_name].keys()):
                cfg = data_config.seed_features_dict[task_name][feat_name]
                if self.is_valid_feat(cfg):
                    feat_configs[(f"__seed__-{task_name}", feat_name)] = cfg
                    if (target_type, feat_name) in feat_configs:
                        _get_or_add_fg(target_type, feat_name).append(
                            (f"__seed__-{task_name}", feat_name)
                        )
                else:
                    _raise_ignore_warning(f"__seed__-{task_name}", feat_name, cfg)

        self.encoder = FeatDictEncoderMultiTask(
            feat_configs,
            feature_groups,
            feat_encode_size,
            use_one_encoder_for_llm,
            with_original_features,
            llm_dim,
        )

        self.node_out_size_dict = defaultdict(lambda: 0)
        for ntype, nt_cfgs in data_config.node_features.items():
            for feat_name in nt_cfgs:
                if add_columns_dim_together:
                    self.node_out_size_dict[ntype] += self.encoder.out_size_dict.get(
                        (ntype, feat_name), 0
                    )
                else:
                    assert (
                        self.node_out_size_dict[ntype]
                        == self.encoder.out_size_dict.get((ntype, feat_name), 0)
                        or self.node_out_size_dict[ntype] == 0
                        or self.encoder.out_size_dict.get((ntype, feat_name), 0) == 0
                    ), f"Node type {ntype} has different output sizes for different features."
                    self.node_out_size_dict[ntype] = self.encoder.out_size_dict.get(
                        (ntype, feat_name), 0
                    )

        self.edge_out_size_dict = defaultdict(lambda: 0)
        for etype, et_cfgs in data_config.edge_features.items():
            for feat_name in et_cfgs:
                if add_columns_dim_together:
                    self.edge_out_size_dict[etype] += self.encoder.out_size_dict.get(
                        (etype, feat_name), 0
                    )
                else:
                    assert (
                        self.edge_out_size_dict[etype]
                        == self.encoder.out_size_dict.get((etype, feat_name), 0)
                        or self.edge_out_size_dict[etype] == 0
                        or self.encoder.out_size_dict.get((etype, feat_name), 0) == 0
                    ), f"Edge type {etype} has different output sizes for different features."
                    self.edge_out_size_dict[etype] = self.encoder.out_size_dict.get(
                        (etype, feat_name), 0
                    )

        self.seed_ctx_out_size_dict = {}
        for task_name in data_config.task_dict.keys():
            self.seed_ctx_out_size_dict[task_name] = 0
            for feat_name in data_config.seed_features_dict[task_name]:
                if add_columns_dim_together:
                    self.seed_ctx_out_size_dict[
                        task_name
                    ] += self.encoder.out_size_dict.get(
                        (f"__seed__-{task_name}", feat_name), 0
                    )
                else:
                    assert (
                        self.seed_ctx_out_size_dict[task_name]
                        == self.encoder.out_size_dict.get(
                            (f"__seed__-{task_name}", feat_name), 0
                        )
                        or self.seed_ctx_out_size_dict[task_name] == 0
                        or self.encoder.out_size_dict.get(
                            (f"__seed__-{task_name}", feat_name), 0
                        )
                        == 0
                    ), f"Seed context for task {task_name} has different output sizes for different features."
                    self.seed_ctx_out_size_dict[task_name] = (
                        self.encoder.out_size_dict.get(
                            (f"__seed__-{task_name}", feat_name), 0
                        )
                    )

        # Check if the node_out_size_dict, edge_out_size_dict, and seed_ctx_out_size_dict have the same values
        if use_one_encoder_for_llm:
            for ntype, out_size in self.node_out_size_dict.items():
                if (
                    out_size
                    != self.node_out_size_dict[list(self.node_out_size_dict.keys())[0]]
                ):
                    raise ValueError(
                        "All node_out_size_dict values must be the same if use_one_encoder_for_llm=True."
                    )
            for etype, out_size in self.edge_out_size_dict.items():
                if (
                    out_size
                    != self.edge_out_size_dict[list(self.edge_out_size_dict.keys())[0]]
                ):
                    raise ValueError(
                        "All edge_out_size_dict values must be the same if use_one_encoder_for_llm=True."
                    )
            for task_name, out_size in self.seed_ctx_out_size_dict.items():
                if (
                    out_size
                    != self.seed_ctx_out_size_dict[
                        list(self.seed_ctx_out_size_dict.keys())[0]
                    ]
                ):
                    raise ValueError(
                        "All seed_ctx_out_size_dict values must be the same if use_one_encoder_for_llm=True."
                    )

    def forward(
        self,
        feat_dict: FeatDict,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Forward function.

        Support both node feature dict and edge feature dict. For edge feature dict,
        the edge type is a string of form "src_type:edge_type:dst_type".
        """
        flat_feat_dict = {}
        for ty, ty_feat_dict in feat_dict.items():
            for feat_name, feat in ty_feat_dict.items():
                flat_feat_dict[(ty, feat_name)] = feat

        flat_feat_dict = self.encoder(flat_feat_dict)

        # Revert to nested dict.
        new_feat_dict = {}
        for ty, ty_feat_dict in feat_dict.items():
            if ty not in new_feat_dict:
                new_feat_dict[ty] = {}
            for feat_name, feat in ty_feat_dict.items():
                # If using orthogonal encoder, we need to convert the output feature name
                if self.encoder.ft2gid[(ty, feat_name)] == 1:
                    feat_name_new = f"Griffin_text_{feat_name}"
                    new_feat_dict[ty][feat_name_new] = flat_feat_dict[(ty, feat_name)]
                else:
                    new_feat_dict[ty][feat_name] = flat_feat_dict[(ty, feat_name)]
        return new_feat_dict

    def __repr__(self):
        super_repr = super().__repr__()
        extra_repr = (
            f"  node_out_size_dict={dict(self.node_out_size_dict)}\n"
            f"  edge_out_size_dict={dict(self.edge_out_size_dict)}\n"
            f"  seed_ctx_out_size_dict={dict(self.seed_ctx_out_size_dict)}\n)"
        )
        return super_repr[:-1] + extra_repr


def _raise_ignore_warning(feat_type, feat_name, cfg):
    logger.warning(f"Ignore feature ({feat_type}, {feat_name}) with dtype {cfg.dtype}.")
