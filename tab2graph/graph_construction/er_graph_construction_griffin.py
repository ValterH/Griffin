from collections import defaultdict
import logging
from pathlib import Path
import shutil
from typing import Optional, Dict, List, Set, Tuple, Any
import numpy as np
import pandas as pd
import pydantic
import torch

from dbinfer_bench import (
    DBBColumnDType,
    DBBColumnID,
    DBBRDBDataset,
    DBBRDBDatasetMeta,
    DBBRDBTask,
    DBBTaskMeta,
    DBBGraphDatasetCreator,
    DBBGraphTaskCreator,
)

from .base import GraphConstruction
from .utils import GriffinLabelFeature

logger = logging.getLogger(__name__)
logger.setLevel("INFO")


class ERGraphGriffinConstructionConfig(pydantic.BaseModel):
    # Whether to construct a relation table as edges.
    # If not, all tables will be constructed as nodes.
    relation_table_as_edge: bool = True
    # multi_enhance means whether we use multi_enhance module
    multi_enhance: bool = False
    # class_node_reverse_edge means whether we add reverse edge for class node to allow message-passing from data nodes to class nodes
    class_node_reverse_edge: bool = False
    # use_time_feature means whether we use time feature as sub_feature nodes
    use_time_feature: bool = False
    # add_fewshot_edges means whether we add few-shot edges for few-shot learning
    add_fewshot_edges: bool = False
    # keep_label_feature means we keep the label feature in seed feature if target node only has foreign key
    keep_label_feature: bool = False


class ERGraphGriffinConstruction(GraphConstruction):

    config_class = ERGraphGriffinConstructionConfig
    name = "r2ne-griffin"

    def build(self, dataset: DBBRDBDataset, output_path: Path):
        table_domain = self.check_relationship(dataset)
        ds_ctor = DBBGraphDatasetCreator(f"{dataset.metadata.dataset_name}-{self.name}")
        self.build_nodes(dataset, table_domain, ds_ctor)
        edge_types, has_nas = self.build_edges(dataset, table_domain, ds_ctor)
        self.build_features(dataset, edge_types, has_nas, ds_ctor)
        self.build_feature_groups(dataset.metadata, edge_types, ds_ctor)
        self.build_task(dataset, edge_types, ds_ctor)
        if self.config.add_fewshot_edges:
            self.build_fewshot(dataset, ds_ctor)
        shutil.rmtree(output_path, ignore_errors=True)
        logger.info("Generating dataset ...")
        ds_ctor.done(output_path)

    """
    Decide whether a table should be constructed as a node or an edge.
        If a table has no primary key and exactly two foreign keys:
            If config.relation_table_as_edge is True:
                Construct the table as an edge.
            Else:
                Construct the table as a node.
        Else:
        (1. If a table has no primary key but not two foreign keys;
         2. If a table has one promary key;)
            Construct the table as a node.

    """

    def check_relationship(self, dataset: DBBRDBDataset) -> Dict[str, str]:
        table_domain = {}
        tblname_to_keys = get_tblname_to_keys_mapping(dataset)
        for tblname, keys in tblname_to_keys.items():
            if len(keys["pk"]) == 0:
                if len(keys["fk"]) == 2 and self.config.relation_table_as_edge:
                    table_domain[tblname] = "edge"
                else:
                    table_domain[tblname] = "node"
            elif len(keys["pk"]) == 1:
                table_domain[tblname] = "node"
            else:
                raise RuntimeError(
                    f"More than one primary key in {tblname}, the"
                    f' primary keys are {keys["pk"]}.'
                )
        logger.info(f"Table domain: {table_domain}")
        return table_domain

    def build_nodes(
        self,
        dataset: DBBRDBDataset,
        table_domain: Dict[str, str],
        ds_ctor: DBBGraphDatasetCreator,
    ):
        logger.info("Building nodes...")
        for tblname, table in dataset.tables.items():
            if table_domain[tblname] == "node":
                col_data = list(table.values())[0]
                nelem = len(col_data)
                ds_ctor.add_nodes(nelem, type=tblname)
                logger.info(f"Add node {tblname} with {nelem} nodes.")

        for task in dataset.tasks:
            target_column = task.metadata.target_column
            target_table = task.metadata.target_table
            if (task.metadata.task_type == "classification") and (
                not self.config.multi_enhance
                or target_column not in dataset.tables[target_table]
            ):
                nelem = task.metadata.num_classes
                ds_ctor.add_nodes(
                    nelem, type=f"{target_table}_{target_column}"
                )

        logger.info("Built all nodes.")

    def build_edges(
        self,
        dataset: DBBRDBDataset,
        table_domain: Dict[str, str],
        ds_ctor: DBBGraphDatasetCreator,
    ) -> Dict[str, List[str]]:
        logger.info("Building edges...")
        edge_types = {}
        has_nas = {}
        tblname_to_keys = get_tblname_to_keys_mapping(dataset)
        fk_to_pk = get_fk_to_pk_mapping(dataset.metadata)

        for tblname, table in dataset.tables.items():
            if table_domain[tblname] == "node":
                # If a table is constructed as a node, then for
                # each of its foreign key, edges will be added
                # between the referred table and itself.
                for fk_col in tblname_to_keys[tblname]["fk"]:
                    dst_pk = fk_to_pk[(tblname, fk_col)]
                    dst = table[fk_col]
                    src = np.arange(len(dst))
                    middle_name = f"{tblname}-{fk_col}"
                    self.add_edges(
                        tblname,
                        src,
                        dst_pk.table,
                        dst,
                        middle_name,
                        ds_ctor,
                        add_timestamp=self.config.relation_table_as_edge,
                    )
                    # Don't need to record the edge type, because
                    # there is no feature appended to it.

            else:
                # If a table is constructed as an edge, then
                # edges will be added between the tables
                # referred by its foreign keys.
                src_col, dst_col = tblname_to_keys[tblname][
                    "fk"
                ]  # Assume there are only two foreign keys.
                src_pk = fk_to_pk[(tblname, src_col)]
                dst_pk = fk_to_pk[(tblname, dst_col)]
                src = table[src_col]
                dst = table[dst_col]
                middle_name = tblname
                etypes, has_na = self.add_edges(
                    src_pk.table, src, dst_pk.table, dst, middle_name, ds_ctor
                )

                # Record the edge type and removed edges for later use.
                edge_type, reverse_edge_type = etypes
                edge_types[tblname] = [edge_type, reverse_edge_type]
                has_nas[tblname] = has_na

        for task in dataset.tasks:
            target_column = task.metadata.target_column
            target_table = task.metadata.target_table
            ntype = f"{target_table}_{target_column}"
            if (task.metadata.task_type == "classification") and (
                not self.config.multi_enhance
                or target_column not in dataset.tables[target_table]
            ):
                # add self-loop for the target table
                unique_src = np.arange(task.metadata.num_classes)
                edge_type = reverse_edge_type = (
                    f"{ntype}:{ntype}-self_loop:{ntype}"
                )
                ds_ctor.add_edges(np.stack([unique_src, unique_src]), edge_type)
                logger.info(
                    f"Add {len(unique_src)} self-loop edges for type {edge_type}."
                )

        logger.info("Built all edges.")
        return edge_types, has_nas

    def build_fewshot(
        self,
        dataset: DBBRDBDataset,
        ds_ctor: DBBGraphDatasetCreator,
    ):
        # Few-shot learning: add edges between the few-shot data nodes and the class nodes.
        logger.info("Building few-shot edges...")
        for task in dataset.tasks:
            task_name = task.metadata.name
            target_column = task.metadata.target_column
            target_table = task.metadata.target_table
            class_ntype = f"{target_table}_{target_column}"
            fewshot_ntype = f"{task_name}_{target_table}_fewshot"
            # Add few-shot nodes.
            nelem = len(task.train_set[target_column])
            ds_ctor.add_nodes(nelem, type=fewshot_ntype) 
            # Add few-shot edges.
            src = np.arange(nelem)
            dst = task.train_set[target_column]
            edge_type = f"{fewshot_ntype}:fewshot:{class_ntype}"
            ds_ctor.add_edges(np.stack([src, dst]), edge_type)
            logger.info(f"Add {len(src)} few-shot edges for type {edge_type}.")
            # Add few-shot node features, only with Griffin_text feature.
            feat_name = "Griffin_text"
            col_meta = {"dtype": 'float'}
            feat = task.train_set[feat_name]
            ds_ctor.add_node_feature(fewshot_ntype, feat_name, "numpy", feat, **col_meta)
            logger.info(f"Add feature {feat_name} for node type {fewshot_ntype}.")
            # Add few-shot node timestamp.
            ts = np.zeros(nelem, dtype=np.int64)
            ds_ctor.add_node_timestamp(fewshot_ntype, ts)
            logger.info(f"Add timestamp for node type {fewshot_ntype}.")
        logger.info("Built few-shots.")

    def add_me_edges(
        self,
        src_type: str,
        src: np.ndarray,
        dst_type: str,
        dst: np.ndarray,
        middle_name: str,
        ds_ctor: DBBGraphDatasetCreator,
        add_timestamp=False,
        add_reverse=True,
    ):
        # Filter out NA keys.
        has_na = np.logical_or((src == -1), (dst == -1))
        n_before = len(src)
        src = src[~has_na]
        dst = dst[~has_na]
        n_after = len(src)
        logger.info(f"Filter out {n_before - n_after} edges with NA keys.")

        if src_type == dst_type:
            # Directly create bi-directional edges instead of adding a new reverse type.
            # Only support add reverse edges.
            edges = np.concatenate([np.stack([src, dst]), np.stack([dst, src])], axis=1)
            edge_type = reverse_edge_type = f"{src_type}:{middle_name}:{dst_type}"
            ds_ctor.add_edges(edges, edge_type)
            logger.info(f"Add {edges.shape[1]} edges for type {edge_type}.")

            if add_timestamp:
                logger.info(f"Add default timestamp for edge type {edge_type}.")
                ts = np.zeros((edges.shape[1],), dtype=np.int64)
                ds_ctor.add_edge_timestamp(edge_type, ts)

        else:
            # Create a separate edge type for reverse edges.
            # * The src_type are the original data nodes, and the dst_type are the class nodes.
            # To align with the seed-type edge direction, we set the edge direction from data nodes to class nodes as original direction,
            # and set the edge direction from class nodes to data nodes as reverse direction.
            edge_type = f"{src_type}:reverse_{middle_name}:{dst_type}"
            ds_ctor.add_edges(np.stack([src, dst]), edge_type)
            logger.info(f"Add {len(src)} edges for type {edge_type}.")

            if add_reverse:
                reverse_edge_type = f"{dst_type}:{middle_name}:{src_type}"
                ds_ctor.add_edges(np.stack([dst, src]), reverse_edge_type)
                logger.info(f"Add {len(src)} edges for type {reverse_edge_type}.")
            else:
                # Add self-loop for the destination type.
                edge_type = reverse_edge_type = (
                    f"{src_type}:{src_type}-self_loop:{src_type}"
                )
                unique_src = np.unique(src)
                ds_ctor.add_edges(np.stack([unique_src, unique_src]), edge_type)
                logger.info(
                    f"Add {len(unique_src)} self-loop edges for type {edge_type}."
                )

            if add_timestamp:
                logger.info(
                    f"Add default timestamp for edge type {edge_type} and {reverse_edge_type}."
                )
                ts = np.zeros(len(src), dtype=np.int64)
                ds_ctor.add_edge_timestamp(edge_type, ts)
                if add_reverse:
                    ds_ctor.add_edge_timestamp(reverse_edge_type, ts)

        return (edge_type, reverse_edge_type), has_na

    def add_edges(
        self,
        src_type: str,
        src: np.ndarray,
        dst_type: str,
        dst: np.ndarray,
        middle_name: str,
        ds_ctor: DBBGraphDatasetCreator,
        add_timestamp=False,
    ):
        # Filter out NA keys.
        has_na = np.logical_or((src == -1), (dst == -1))
        n_before = len(src)
        src = src[~has_na]
        dst = dst[~has_na]
        n_after = len(src)
        logger.info(f"Filter out {n_before - n_after} edges with NA keys.")

        if src_type == dst_type:
            # Directly create bi-directional edges instead of adding a new reverse type.
            edges = np.concatenate([np.stack([src, dst]), np.stack([dst, src])], axis=1)
            edge_type = reverse_edge_type = f"{src_type}:{middle_name}:{dst_type}"
            ds_ctor.add_edges(edges, edge_type)
            logger.info(f"Add {edges.shape[1]} edges for type {edge_type}.")

            if add_timestamp:
                logger.info(f"Add default timestamp for edge type {edge_type}.")
                ts = np.zeros((edges.shape[1],), dtype=np.int64)
                ds_ctor.add_edge_timestamp(edge_type, ts)

        else:
            # Create a separate edge type for reverse edges.
            edge_type = f"{src_type}:{middle_name}:{dst_type}"
            ds_ctor.add_edges(np.stack([src, dst]), edge_type)
            logger.info(f"Add {len(src)} edges for type {edge_type}.")

            reverse_edge_type = f"{dst_type}:reverse_{middle_name}:{src_type}"
            ds_ctor.add_edges(np.stack([dst, src]), reverse_edge_type)
            logger.info(f"Add {len(src)} edges for type {reverse_edge_type}.")

            if add_timestamp:
                logger.info(
                    f"Add default timestamp for edge type {edge_type} and {reverse_edge_type}."
                )
                ts = np.zeros(len(src), dtype=np.int64)
                ds_ctor.add_edge_timestamp(edge_type, ts)
                ds_ctor.add_edge_timestamp(reverse_edge_type, ts)

        return (edge_type, reverse_edge_type), has_na

    def build_features(
        self,
        dataset: DBBRDBDataset,
        edge_types: Dict[str, List[str]],
        has_nas: Dict[str, np.ndarray],
        ds_ctor: DBBGraphDatasetCreator,
    ):
        logger.info("Building features...")
        for table_meta in dataset.metadata.tables:
            table_name = table_meta.name
            table = dataset.tables[table_name]
            for col_meta in table_meta.columns:
                name = col_meta.name
                feat = table[name]
                # Skip primary key and foreign key.
                if (
                    col_meta.dtype == DBBColumnDType.primary_key
                    or col_meta.dtype == DBBColumnDType.foreign_key
                ):
                    continue
                # Add feature to Creator.
                col_meta = dict(col_meta)
                col_meta.pop("name")
                is_time_column = name == table_meta.time_column
                if table_name in edge_types:
                    feat = feat[~has_nas[table_name]]
                    etype, rev_etype = edge_types[table_name]
                    if etype == rev_etype:
                        # Reverse edges are added as bi-directional edges.
                        feat = np.concatenate([feat, feat], axis=0)
                        if is_time_column:
                            ds_ctor.add_edge_timestamp(etype, feat)
                            logger.info(f"Add timestamp for edge type {etype}.")
                        else:
                            ds_ctor.add_edge_feature(
                                etype, name, "numpy", feat, **col_meta
                            )
                            logger.info(f"Add feature {name} for edge type {etype}.")
                    else:
                        # Reverse edges belong to a different type.
                        if is_time_column:
                            ds_ctor.add_edge_timestamp(etype, feat)
                            ds_ctor.add_edge_timestamp(rev_etype, feat)
                            logger.info(
                                f"Add timestamp for edge type {etype} and {rev_etype}."
                            )
                        else:
                            ds_ctor.add_edge_feature(
                                etype, name, "numpy", feat, **col_meta
                            )
                            ds_ctor.add_edge_feature(
                                rev_etype, name, "numpy", feat, **col_meta
                            )
                            logger.info(
                                f"Add feature {name} for edge type {etype} and {rev_etype}."
                            )
                else:
                    ntype = table_name
                    if is_time_column:
                        ds_ctor.add_node_timestamp(ntype, feat)
                        logger.info(f"Add timestamp for node type {ntype}.")
                    else:
                        ds_ctor.add_node_feature(ntype, name, "numpy", feat, **col_meta)
                        logger.info(f"Add feature {name} for node type {ntype}.")

        griffin_label_feature = GriffinLabelFeature(dataset_name=dataset.metadata.dataset_name)
        for task in dataset.tasks:
            target_column = task.metadata.target_column
            target_table = task.metadata.target_table
            if not self.config.multi_enhance or target_column not in dataset.tables[target_table]:
                task_name, task_type = task.metadata.name, task.metadata.task_type
                if task_type != "classification":
                    # TODO: Support other task types.
                    continue
                # table_name = task.metadata.target_table
                # column_name = task.metadata.target_column
                ntype = f"{target_table}_{target_column}"
                col_meta = dict(task.metadata.column_dict[target_column])
                # It seems the num_categories will be num_classes + 1. Now fix it.
                # col_meta["num_categories"] = task.metadata.num_classes
                col_meta.pop("name")
                col_meta["dtype"] = 'float'
                # Seems the label is already remapped to numbers.
                # Assert the label is already remapped to numbers.
                assert task.train_set[target_column].dtype == np.int64
                # Compute feature with LLM, also remap the categories to numbers
                label_set = np.concatenate([task.train_set[target_column], task.validation_set[target_column], task.test_set[target_column]])
                _, lable_categories = pd.factorize(label_set, sort=True, use_na_sentinel=False)
                # Find the category mapping on the target column
                if hasattr(task.metadata.column_dict[target_column], 'category_mapping'):
                    category_mapping = task.metadata.column_dict[target_column].category_mapping
                    category_mapping = {int(k): str(v) for k, v in category_mapping.items()}
                else:
                    category_mapping = None
                # Update the category mapping with keys to integers
                feat = griffin_label_feature.encode_with_template(
                    lable_categories,
                    task_name,
                    target_column,
                    category_mapping
                )
                # task.train_set[target_column] = remapped_label[:len(task.train_set[target_column])]
                # task.validation_set[target_column] = remapped_label[len(task.train_set[target_column]):len(task.train_set[target_column])+len(task.validation_set[target_column])]
                # task.test_set[target_column] = remapped_label[len(task.train_set[target_column])+len(task.validation_set[target_column]):]
                # feat = np.arange(task.metadata.num_classes)
                ds_ctor.add_node_feature(ntype, f"Griffin_text_{target_column}", "numpy", feat, **col_meta)
                timestamp_feat = np.zeros(len(feat), dtype=np.int64)
                ds_ctor.add_node_timestamp(ntype, timestamp_feat)

        logger.info("Built all features.")

    def build_feature_groups(
        self,
        metadata: DBBRDBDatasetMeta,
        edge_types: Dict[str, List[str]],
        ds_ctor: DBBGraphDatasetCreator,
    ):
        logger.info("Building feature groups...")
        added_edge_cols = set()
        if metadata.column_groups is not None:
            for column_group in metadata.column_groups:
                feat_group = []
                for col in column_group:
                    table_name, column_name = col.table, col.column
                    if table_name in edge_types:
                        feat_group.append(
                            ("edge", edge_types[table_name][0], column_name)
                        )
                        feat_group.append(
                            ("edge", edge_types[table_name][1], column_name)
                        )
                        added_edge_cols.add((table_name, column_name))
                    else:
                        feat_group.append(("node", table_name, column_name))
                ds_ctor.add_feature_group(feat_group)

        # Create feature group for reverse edges.
        for table_meta in metadata.tables:
            table_name = table_meta.name
            if table_name in edge_types:
                for col_schema in table_meta.columns:
                    col_name = col_schema.name
                    if (table_name, col_name) in added_edge_cols:
                        continue
                    if (
                        col_schema.dtype == DBBColumnDType.primary_key
                        or col_schema.dtype == DBBColumnDType.foreign_key
                    ):
                        continue
                    if col_name == table_meta.time_column:
                        # Time columns are added as timestamp instead of features.
                        continue
                    ds_ctor.add_feature_group(
                        [
                            ("edge", edge_types[table_name][0], col_name),
                            ("edge", edge_types[table_name][1], col_name),
                        ]
                    )

        logger.info("Built feature groups.")

    def build_task(
        self,
        dataset: DBBRDBDataset,
        edge_types: Dict[str, List[str]],
        ds_ctor: DBBGraphDatasetCreator,
    ):
        logger.info("Building task...")
        for task in dataset.tasks:
            logger.info(f"\tProcessing task {task.metadata.name} ...")
            ds_ctor.add_task(self.build_one_task(task, dataset, edge_types))
        logger.info("Built task.")

    def build_one_task(
        self,
        task: DBBRDBTask,
        dataset: DBBRDBDataset,
        edge_types: Optional[Dict[str, List[str]]],
    ) -> DBBGraphTaskCreator:
        if task.metadata.task_type == 'retrieval':
            task_ctor = DBBGraphTaskCreator(task.metadata.name)
            task_ctor.add_meta(task.metadata)
            tgt_tbl = task.metadata.target_table
            # Build seeds from pk/fk.
            train_seeds, val_seeds, test_seeds = [], [], []
            task_pk, task_fk = get_task_keys(task.metadata)
            key_cols = {}
            for col in task_pk + task_fk:
                key_cols[col] = (
                    task.train_set[col],
                    task.validation_set[col],
                    task.test_set[col],
                )

            num_keys = len(key_cols)
            assert num_keys == 2
            key1, key2 = list(key_cols.keys())
            fk_to_pk = get_fk_to_pk_mapping(dataset.metadata, pk_map_to_pk=True)
            key1_type = fk_to_pk[(tgt_tbl, key1)].table
            key2_type = fk_to_pk[(tgt_tbl, key2)].table
            if tgt_tbl in edge_types:
                # Edge type exists.
                seed_type = target_type = edge_types[tgt_tbl][0]
                src_type, _, dst_type = target_type.split(':')
                if src_type == key1_type and dst_type == key2_type:
                    src_key, dst_key = key1, key2
                elif src_type == key2_type and dst_type == key1_type:
                    src_key, dst_key = key2, key1
                else:
                    raise ValueError(f"Invalid key cols in task table: {key_cols.keys()}")
                logger.info(f"Create graph task: prediction over edges of type {seed_type}.")
            else:
                # Edge type does not exist.
                fk_to_pk = get_fk_to_pk_mapping(dataset.metadata, pk_map_to_pk=True)
                src_key, dst_key = key1, key2
                src_type = fk_to_pk[(tgt_tbl, src_key)].table
                dst_type = fk_to_pk[(tgt_tbl, dst_key)].table
                seed_type = target_type = f'{src_type}:{tgt_tbl}:{dst_type}'
                logger.info(f"Create graph task: prediction over node-pair of type {seed_type}.")

            seed_col_names = [src_key, dst_key]
            train_src_seeds, val_src_seeds, test_src_seeds = key_cols[src_key]
            train_dst_seeds, val_dst_seeds, test_dst_seeds = key_cols[dst_key]
            train_seeds = np.stack([train_src_seeds, train_dst_seeds]).T
            val_seeds = np.stack([val_src_seeds, val_dst_seeds]).T
            test_seeds = np.stack([test_src_seeds, test_dst_seeds]).T

            if -1 in train_seeds or -1 in val_seeds or -1 in test_seeds:
                raise ValueError("N/A seeds (-1) are detected!")
            task_ctor.set_seed_type(seed_type)
            task_ctor.set_target_type(target_type)
            task_ctor.set_seeds(train_seeds, val_seeds, test_seeds, type=seed_type)
            # Build labels.
            tgt_col = task.metadata.target_column
            # Retrieval task.
            assert len(seed_col_names) > 1
            target_seed_idx = seed_col_names.index(tgt_col)
            task_ctor.add_extra_field('target_seed_idx', target_seed_idx)
            key_pred_label = task.metadata.key_prediction_label_column
            task_ctor.set_labels(
                None,
                task.validation_set[key_pred_label],
                task.test_set[key_pred_label],
                type=seed_type
            )
            key_pred_query_idx = task.metadata.key_prediction_query_idx_column
            task_ctor.add_item(
                key_pred_query_idx,
                None,
                task.validation_set[key_pred_query_idx],
                task.test_set[key_pred_query_idx],
                type=seed_type
            )

            # Other item set attributes in task.
            for col_meta in task.metadata.columns:
                col = col_meta.name
                dtype = col_meta.dtype
                # Skip primary key and foreign key.
                if dtype == DBBColumnDType.primary_key or \
                    dtype == DBBColumnDType.foreign_key:
                    continue
                # Skip labels.
                if col == tgt_col:
                    continue
                col_meta = dict(col_meta)
                col_meta.pop('name')
                if col == task.metadata.time_column:
                    task_ctor.set_seed_timestamp(
                        task.train_set[col],
                        task.validation_set[col],
                        task.test_set[col],
                        type=seed_type
                    )
                else:
                    task_ctor.add_seed_feature(
                        col,
                        task.train_set[col],
                        task.validation_set[col],
                        task.test_set[col],
                        type=seed_type,
                        **col_meta
                    )

            return task_ctor
        elif task.metadata.task_type == "regression":
            task_ctor = DBBGraphTaskCreator(task.metadata.name)
            task_ctor.add_meta(task.metadata)
            tgt_col = task.metadata.target_column
            tgt_tbl = task.metadata.target_table

            # Build seeds from pk/fk.
            train_seeds, val_seeds, test_seeds = [], [], []
            task_pk, task_fk = get_task_keys(task.metadata)
            key_cols = {}
            for col in task_pk + task_fk:
                key_cols[col] = (
                    task.train_set[col],
                    task.validation_set[col],
                    task.test_set[col],
                )
            
            num_keys = len(key_cols)
            assert num_keys > 0
            if num_keys == 1:
                target_type = tgt_tbl
                if len(task_pk) == 1:
                    # Task table has one primary key. The corresponding graph task is node
                    # prediction.
                    seed_type = tgt_tbl
                    logger.info(f"Create graph task: prediction over nodes of type {seed_type}.")
                else:
                    # Task table has one foreign key. The corresponding graph task is node
                    # prediction over the referred table.
                    fk_to_pk = get_fk_to_pk_mapping(dataset.metadata)
                    referred_table = fk_to_pk[(tgt_tbl, task_fk[0])].table
                    seed_type = referred_table
                    logger.info(f"Create graph task: prediction over nodes of type {seed_type}.")
                train_seeds, val_seeds, test_seeds = list(key_cols.values())[0]
            elif len(task_pk) == 1:
                # More than one keys in the task table but one of them is primary key.
                # Convert it to a node prediction task.
                target_type = seed_type = tgt_tbl
                train_seeds, val_seeds, test_seeds = key_cols[task_pk[0]]
                logger.info(f"Create graph task: prediction over node of type {seed_type}.")
            else:
                raise ValueError("Relations beyond binary are not supported yet.")
            if -1 in train_seeds or -1 in val_seeds or -1 in test_seeds:
                raise ValueError("N/A seeds (-1) are detected!")
            task_ctor.set_seed_type(seed_type)
            task_ctor.set_target_type(target_type)
            task_ctor.set_seeds(train_seeds, val_seeds, test_seeds, type=seed_type)
            
            task_ctor.set_labels(
                task.train_set[tgt_col],    
                task.validation_set[tgt_col],
                task.test_set[tgt_col],
                type=seed_type
            )

            # Other item set attributes in task.
            for col_meta in task.metadata.columns:
                col = col_meta.name
                dtype = col_meta.dtype
                # Skip primary key and foreign key.
                if dtype == DBBColumnDType.primary_key or \
                    dtype == DBBColumnDType.foreign_key:
                    continue
                # Skip labels.
                if col == tgt_col:
                    if self.config.keep_label_feature:
                        if len(task_pk) == 0 and len(task_fk) == 1:
                            # If the target node only has one foreign key, we keep the label feature in seed feature.
                            pass
                        else:
                            continue
                    else:
                        continue
                col_meta = dict(col_meta)
                col_meta.pop('name')
                if col == task.metadata.time_column:
                    task_ctor.set_seed_timestamp(
                        task.train_set[col],
                        task.validation_set[col],
                        task.test_set[col],
                        type=seed_type
                    )
                else:
                    task_ctor.add_seed_feature(
                        col,
                        task.train_set[col],
                        task.validation_set[col],
                        task.test_set[col],
                        type=seed_type,
                        **col_meta
                    )
            
            return task_ctor

        task_ctor = DBBGraphTaskCreator(task.metadata.name)
        # Update task type to Griffin.
        assert task.metadata.task_type == "classification", "We only support classification task."
        assert task.metadata.evaluation_metric == "accuracy" or task.metadata.evaluation_metric == "auroc" or task.metadata.evaluation_metric == "logloss", "We only support accuracy & auroc & logloss metric."
        task.metadata.task_type = "retrieval"
        if task.metadata.evaluation_metric == "accuracy":
            task.metadata.evaluation_metric = "hr@1"
        elif task.metadata.evaluation_metric == "auroc":
            task.metadata.evaluation_metric = "retrieval_auroc"
        elif task.metadata.evaluation_metric == "logloss":
            task.metadata.evaluation_metric = "retrieval_logloss"
        task.metadata.key_prediction_label_column = "label"
        task.metadata.key_prediction_query_idx_column = "query_idx"
        task_ctor.add_meta(task.metadata)
        task_ctor.add_extra_field("num_classes", task.metadata.num_classes)

        tgt_tbl = task.metadata.target_table

        # Build seeds from pk/fk.
        train_seeds, val_seeds, test_seeds = [], [], []
        task_pk, task_fk = get_task_keys(task.metadata)
        key_cols = {}
        if len(task_pk) == 1:
            for col in task_pk:
                key_cols[col] = (
                    task.train_set[col],
                    task.validation_set[col],
                    task.test_set[col],
                )
        else:
            logger.warning("No primary key in the task table.")
            for col in task_pk + task_fk:
                key_cols[col] = (
                    task.train_set[col],
                    task.validation_set[col],
                    task.test_set[col],
                )

        assert len(task_pk) == 1 or len(task_fk) == 1

        target_type = tgt_tbl
        if len(task_pk) == 1:
            # Task table has one primary key. The corresponding graph task is node
            # prediction.
            seed_type = tgt_tbl
            logger.info(
                f"Create graph task: prediction over nodes of type {seed_type}."
            )
        else:
            # Task table has one foreign key. The corresponding graph task is node
            # prediction over the referred table.
            fk_to_pk = get_fk_to_pk_mapping(dataset.metadata)
            referred_table = fk_to_pk[(tgt_tbl, task_fk[0])].table
            seed_type = referred_table
            logger.info(
                f"Create graph task: prediction over nodes of type {seed_type}."
            )
        # Update to Griffin task.
        tgt_col = task.metadata.target_column
        src_key = seed_type
        dst_key = f"{tgt_tbl}_{tgt_col}"
        seed_type = f"{src_key}:{src_key}-{dst_key}:{dst_key}"
        seed_col_names = [src_key, dst_key]

        def _extend_seeds(pos_src, pos_dst, num_classes):
            neg_src = np.repeat(pos_src, num_classes - 1)
            neg_dst = np.tile(np.arange(num_classes - 1), len(pos_src))
            pos_dst_aug = np.repeat(pos_dst, num_classes - 1)
            neg_dst = np.where(neg_dst == pos_dst_aug, num_classes - 1, neg_dst)
            query_idx = np.concatenate(
                [
                    np.arange(len(pos_src)),
                    np.repeat(np.arange(len(pos_src)), num_classes - 1),
                ]
            )
            labels = np.concatenate(
                [
                    np.ones(len(pos_src), dtype=np.int64),
                    np.zeros(len(neg_src), dtype=np.int64),
                ]
            )
            src = np.concatenate([pos_src, neg_src])
            dst = np.concatenate([pos_dst, neg_dst])
            return src, dst, query_idx, labels

        train_src_seeds, val_src_seeds, test_src_seeds = list(key_cols.values())[0]
        train_dst_seeds, val_dst_seeds, test_dst_seeds = (
            task.train_set[tgt_col],
            task.validation_set[tgt_col],
            task.test_set[tgt_col],
        )
        val_src_seeds, val_dst_seeds, val_query_idx, val_labels = _extend_seeds(
            val_src_seeds, val_dst_seeds, task.metadata.num_classes
        )
        test_src_seeds, test_dst_seeds, test_query_idx, test_labels = _extend_seeds(
            test_src_seeds, test_dst_seeds, task.metadata.num_classes
        )

        train_seeds = np.stack([train_src_seeds, train_dst_seeds]).T
        val_seeds = np.stack([val_src_seeds, val_dst_seeds]).T
        test_seeds = np.stack([test_src_seeds, test_dst_seeds]).T

        task_ctor.set_seed_type(seed_type)
        task_ctor.set_target_type(target_type)
        task_ctor.set_seeds(train_seeds, val_seeds, test_seeds, type=seed_type)

        # Build labels.
        if task.metadata.task_type == "retrieval":
            # Retrieval task.
            assert len(seed_col_names) > 1
            target_seed_idx = seed_col_names.index(f"{tgt_tbl}_{tgt_col}")
            task_ctor.add_extra_field("target_seed_idx", target_seed_idx)
            key_pred_label = task.metadata.key_prediction_label_column
            task_ctor.set_labels(
                None,
                val_labels,
                test_labels,
                type=seed_type,
            )
            key_pred_query_idx = task.metadata.key_prediction_query_idx_column
            task_ctor.add_item(
                key_pred_query_idx,
                None,
                val_query_idx,
                test_query_idx,
                type=seed_type,
            )
        else:
            AssertionError("Unsupported task type.")
            # Classification, regression tasks.
            task_ctor.set_labels(
                task.train_set[tgt_col],
                task.validation_set[tgt_col],
                task.test_set[tgt_col],
                type=seed_type,
            )

        # Other item set attributes in task.
        def _extend_seed(seed, num_classes):
            # If the seed is multi-dimensional, repeat the seed should also be multi-dimensional.
            return np.concatenate([seed, np.repeat(seed, num_classes - 1, axis=0)])


        for col_meta in task.metadata.columns:
            col = col_meta.name
            dtype = col_meta.dtype
            # Skip primary key and foreign key.
            if (
                dtype == DBBColumnDType.primary_key
                or dtype == DBBColumnDType.foreign_key
            ):
                continue
            # Skip labels.
            if col == tgt_col:
                if self.config.keep_label_feature:
                    if len(task_pk) == 0 and len(task_fk) == 1:
                        # If the target node only has one foreign key, we keep the label feature in seed feature.
                        pass
                    else:
                        continue
                else:
                    continue
            col_meta = dict(col_meta)
            col_meta.pop("name")
            if col == task.metadata.time_column:
                task_ctor.set_seed_timestamp(
                    task.train_set[col],
                    _extend_seed(
                        task.validation_set[col], num_classes=task.metadata.num_classes
                    ),
                    _extend_seed(
                        task.test_set[col], num_classes=task.metadata.num_classes
                    ),
                    type=seed_type,
                )
            else:
                # TODO: It is possible the seed_type = target_type, which means the features on node should not be used.
                # We need add a check here. Should be refined.
                task_ctor.add_seed_feature(
                    col,
                    task.train_set[col],
                    _extend_seed(
                        task.validation_set[col], num_classes=task.metadata.num_classes
                    ),
                    _extend_seed(
                        task.test_set[col], num_classes=task.metadata.num_classes
                    ),
                    type=seed_type,
                    **col_meta,
                )

        return task_ctor


def get_tblname_to_keys_mapping(
    dataset: DBBRDBDataset,
) -> Dict[str, Dict[str, List[str]]]:
    tblname_to_keys = defaultdict(lambda: {"pk": set(), "fk": list()})
    for r in dataset.metadata.relationships:
        tblname_to_keys[r.fk.table]["fk"].append(r.fk.column)
        tblname_to_keys[r.pk.table]["pk"].add(r.pk.column)
    for keys in tblname_to_keys.values():
        keys["pk"] = list(keys["pk"])
    task_pks = get_all_task_pks(dataset)
    for tblname, colname in task_pks:
        if tblname not in tblname_to_keys:
            tblname_to_keys[tblname]["pk"] = [colname]
    for tbl_dict in dataset.metadata.primary_keys:
        tblname = tbl_dict["table"]
        colname = tbl_dict["column"]
        if "pk" in tblname_to_keys[tblname]:
            pass
        else:
            tblname_to_keys[tblname]["pk"] = [colname]
    return tblname_to_keys


def get_fk_to_pk_mapping(
    metadata: DBBRDBDatasetMeta, pk_map_to_pk: Optional[bool] = False
) -> Dict[Tuple[str, str], DBBColumnID]:
    fk_to_pk = {}
    for r in metadata.relationships:
        fk_to_pk[(r.fk.table, r.fk.column)] = r.pk
    if pk_map_to_pk:
        for r in metadata.relationships:
            fk_to_pk[(r.pk.table, r.pk.column)] = r.pk
    return fk_to_pk


def get_task_keys(metadata: DBBTaskMeta):
    task_pk, task_fk = [], []
    for col_meta in metadata.columns:
        col_name = col_meta.name
        if col_meta.dtype == DBBColumnDType.primary_key:
            task_pk.append(col_name)
        elif col_meta.dtype == DBBColumnDType.foreign_key:
            task_fk.append(col_name)
    return task_pk, task_fk


def get_all_task_pks(dataset: DBBRDBDataset):
    task_pks = set()
    for task in dataset.tasks:
        task_pk, _ = get_task_keys(task.metadata)
        if len(task_pk) == 1:
            tgt_tbl = task.metadata.target_table
            task_pks.add((tgt_tbl, task_pk[0]))
    return task_pks

