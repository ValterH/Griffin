from typing import Tuple, Dict, Optional, List
import numpy as np
import pandas as pd
import pydantic
import logging
import yaml
import hashlib
from collections import defaultdict
from dbinfer_bench import DBBColumnDType

from ...device import DeviceInfo
from ... import datetime_utils
from .base import (
    RDBTransform,
    rdb_transform,
    ColumnData,
    RDBData,
    is_task_table,
)
from sentence_transformers import SentenceTransformer
import os
from tqdm import tqdm
import time
from urllib.parse import quote_plus


logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


__SPLIT_MAX_SIZE__ = 10000000
__SPLIT_NUM__ = 10


class GriffinFeatureSplitColumnConfig(pydantic.BaseModel):
    # LLM_name: str = "ST"
    LLM_name: str = "nomic"
    LLM_batch_size: int = 64
    LLM_root_dir: str = "cache_data/model"
    device: DeviceInfo
    dataset_name: str
    text_backup_dir: str = "cache_text/"
    multi_gpu: bool = True
    # if keep_original_columns is True, the original columns will be kept in the output,
    # except for the columns representing time but in category dtype, and text columns
    keep_original_columns: bool = False
    allow_no_template: bool = False


@rdb_transform
class GriffinFeature_SplitColumn(RDBTransform):
    """Griffin Feature by Split Column:

    Criterion:
    For each table, merge the columns to text and encode the text to feature using LLM.
    """

    config_class = GriffinFeatureSplitColumnConfig
    name = "griffin_feature_split_column"

    def __init__(self, config: GriffinFeatureSplitColumnConfig):
        super().__init__(config)
        self.LLM_name = config.LLM_name
        self.LLM_batch_size = config.LLM_batch_size
        self.LLM_root_dir = config.LLM_root_dir
        self.dataset_name = config.dataset_name
        self.text_backup_dir = config.text_backup_dir + self.dataset_name + "/"
        self.device = config.device
        self.multi_gpu = config.multi_gpu
        # read TEMPLATE from yaml file by pydantic
        TEMPLATE_PATH = os.path.join("configs/transform/griffin_feature_template.yaml")
        with open(TEMPLATE_PATH, "r") as f:
            TEMPLATE = yaml.safe_load(f)
        self.template = TEMPLATE.get(config.dataset_name, {}).get("template", {})
        self.modify_function_dict = TEMPLATE.get(config.dataset_name, {}).get(
            "Modify_function", {}
        )
        if self.LLM_name == "ST":
            self.model = SentenceTransformer(
                "multi-qa-distilbert-cos-v1",
                device=self.device.gpu_devices[0],
                cache_folder=self.LLM_root_dir,
            )
            self._encode = self._ST_encode
            self.LLM_dim = 768
        elif self.LLM_name == "nomic":
            self.LLM_dim = 768
            self.model = SentenceTransformer(
                "nomic-ai/nomic-embed-text-v1.5",
                device=self.device.gpu_devices[0],
                cache_folder=self.LLM_root_dir,
                trust_remote_code=True,
                truncate_dim=self.LLM_dim,
            )
            self._encode = self._nomic_encode

    def _encode(self, texts, *args, **kwargs):
        raise NotImplementedError("Not define llm encoder yet")

    def _ST_encode(self, texts, *args, **kwargs):
        if self.multi_gpu and len(self.device.gpu_devices) > 1:
            start_time = time.time()
            pool = self.model.start_multi_process_pool()
            embeddings = self.model.encode_multi_process(
                texts,
                batch_size=self.LLM_batch_size,
                pool=pool,
            )
            self.model.stop_multi_process_pool(pool)
            end_time = time.time()
            logger.info(f"Time for encoding: {end_time - start_time}")
        else:
            embeddings = self.model.encode(
                texts,
                batch_size=self.LLM_batch_size,
                show_progress_bar=True,
            )
        return embeddings

    def _nomic_encode(self, texts, *args, **kwargs):
        with_dict_accelerate = kwargs.get("with_dict_accelerate", False)
        if with_dict_accelerate:
            # First try 10000 samples to get whether the speedup is worth it
            text_hashes_sample = [
                hashlib.md5(text.encode("utf-8")).hexdigest() for text in texts[:10000]
            ]
            if len(set(text_hashes_sample)) > 0.8 * len(text_hashes_sample):
                logger.info(
                    f"The speedup is not worth it. Ratio of unique texts: {len(set(text_hashes_sample)) / len(text_hashes_sample)}"
                )
                with_dict_accelerate = False
            else:
                logger.info(
                    f"The speedup is worth it. Ratio of unique texts: {len(set(text_hashes_sample)) / len(text_hashes_sample)}"
                )
                with_dict_accelerate = True
        if with_dict_accelerate:
            # Create dictionary mapping text to index and get unique texts
            text_to_idx = {}
            unique_texts = []
            indices = np.empty(len(texts), dtype=np.int32)

            # Single pass through texts for both unique collection and index mapping
            for i, text in enumerate(tqdm(texts)):
                idx = text_to_idx.get(text)
                if idx is None:
                    idx = len(unique_texts)
                    text_to_idx[text] = idx
                    unique_texts.append(text)
                indices[i] = idx

            logger.info(
                f"Original texts: {len(texts)}, Unique texts: {len(unique_texts)}"
            )

            # Generate embeddings only for unique texts
            unique_embeddings = self.model.encode(
                unique_texts,
                batch_size=self.LLM_batch_size,
                show_progress_bar=True,
                prompt="clustering: ",
            ).astype("float32")

            # Map back to original order
            embeddings = unique_embeddings[indices]
        else:
            embeddings = self.model.encode(
                texts,
                batch_size=self.LLM_batch_size,
                show_progress_bar=True,
                prompt="clustering: ",
            ).astype("float32")
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
        return embeddings

    def fit(self, rdb_data: RDBData, device: DeviceInfo):
        self.columns_to_keep = defaultdict(list)
        # self.columns_to_extend_dim = defaultdict(list)
        self.columns_to_merge_to_text = defaultdict(list)
        self.tbl_data_length = defaultdict(int)

        self.tbl_col_name = {}

        for tbl_name, tbl in rdb_data.tables.items():
            self.tbl_col_name[tbl_name] = {}
            for col_name, _ in tbl.items():
                self.tbl_col_name[tbl_name][col_name] = (
                    f"task table: {tbl_name.split(':')[-1]}, column: {col_name}"
                    if tbl_name.startswith("__task__")
                    else f"table: {tbl_name}, column: {col_name}"
                )
            self.tbl_col_name[tbl_name]["default_value"] = (
                f"task table: {tbl_name.split(':')[-1]}, column: default_value"
                if tbl_name.startswith("__task__")
                else f"table: {tbl_name}, column: default_value"
            )

        for tbl_name, tbl in rdb_data.tables.items():
            if tbl_name in self.template:
                logger.info(f"Template for {tbl_name} found.")
                template_for_tbl = self.template[tbl_name]
            elif tbl_name.startswith("__task__"):
                # For task table, the template is not necessary
                # First, find the template for target table.
                target_tbl_name = tbl_name.split(":")[-1]
                target_col_name = tbl_name.split(":")[1].split("-")[-1]
                target_tbl_template = self.template[target_tbl_name]
                template_for_tbl = {}
                for col_name in tbl.keys():
                    flag, template = generate_target_template(
                        col_name, target_tbl_template
                    )
                    if flag:
                        template_for_tbl[col_name] = template
                self.template[tbl_name] = template_for_tbl
                # Add a default value template
                self.template[tbl_name]["default_value"] = f"Predicting the column {target_col_name} with default value for {target_tbl_name}."
            elif self.config.allow_no_template:
                self.template[tbl_name] = {}
                for col_name, col in tbl.items():
                    # Only generate template for non-primary-key, non-foreign-key, non-float columns
                    if not col.metadata["dtype"] in [
                        DBBColumnDType.primary_key,
                        DBBColumnDType.foreign_key,
                        DBBColumnDType.float_t,
                    ]:
                        # For timestamp column, find the number of unique values on random 1000 samples
                        if col.metadata.get("is_time_column", False):
                            unique_values = len(pd.unique(col.data[:1000]))
                            if unique_values <= 1:
                                pass
                            else:
                                self.template[tbl_name][col_name] = f"The {col_name} is {{DAYOFWEEK}}, {{MONTH}} {{DAY}}, {{YEAR}}"
                        else:
                            safe_f_col_name = col_name.replace(".", "_")
                            self.template[tbl_name][col_name] = f"The {col_name} is {{{safe_f_col_name}}}"
                if self.template[tbl_name] == {}:
                    self.template[tbl_name] = {"default_value": f"The default value of {col_name}."}
                template_for_tbl = self.template[tbl_name]
            else:
                raise ValueError(f"Template for {tbl_name} not found.")
            # Find the columns to keep, extend_dim, and merge_to_text
            with_meaningful_float_columns = False
            for col_name, col in tbl.items():
                if col.metadata["dtype"] in [
                    DBBColumnDType.primary_key,
                    DBBColumnDType.foreign_key,
                ]:
                    self.columns_to_keep[tbl_name].append(col_name)
                elif col.metadata.get("is_time_column", False):
                    logger.info(f"Column {col_name} is time column.")
                    self.columns_to_keep[tbl_name].append(col_name)
                    self.columns_to_merge_to_text[tbl_name].append(col_name)
                elif is_task_table(tbl_name) and col.metadata.get(
                    "is_target_column", False
                ):
                    logger.info(f"Column {col_name} is target column.")
                    self.columns_to_keep[tbl_name].append(col_name)
                    # # Also add the column to merge_to_text
                    # self.columns_to_merge_to_text[tbl_name].append(col_name)
                elif col.data.ndim > 1:
                    # Currently, only support 1D vector embeddings
                    AssertionError(f"Column {col_name} has dimension > 1.")
                    # for vector embeddings, will extend the dimension to LLm
                elif col.metadata["dtype"] == DBBColumnDType.float_t:
                    logger.info(f"Column {col_name} is float type.")
                    self.columns_to_keep[tbl_name].append(col_name)
                    with_meaningful_float_columns = True
                else:
                    logger.info(f"Fitting Griffin Feature for {col_name}.")
                    self.columns_to_merge_to_text[tbl_name].append(col_name)
                # Check if the data length is consistent
                if not self.tbl_data_length[tbl_name]:
                    self.tbl_data_length[tbl_name] = len(col.data)
                else:
                    assert self.tbl_data_length[tbl_name] == len(
                        col.data
                    ), f"For table {tbl_name}, data length mismatch: {self.tbl_data_length[tbl_name]} vs {len(col.data)}"
                if col_name in self.columns_to_merge_to_text[tbl_name]:
                    # Check if the template for the column is found
                    if ("YEAR(" in col_name or "MONTH(" in col_name or "DAY(" in col_name or "DAYOFWEEK(" in col_name or "TIMESTAMP(" in col_name):
                        logger.info(f"Template for category generated by time column {col_name} do not need to be generated.")
                        self.columns_to_merge_to_text[tbl_name].remove(col_name)
                    elif col_name in template_for_tbl:
                        logger.info(f"Template for {col_name} found.")
                        logger.info(f"Template: {template_for_tbl[col_name]}")
                        # Give a sample in using TEMPLATE
                        insert_dict = generate_value_dicts(
                            col_name,
                            col.metadata["dtype"],
                            col.data[0:1],
                            self.modify_function_dict.get(tbl_name, {}),
                        )
                        insert_dict_f_safe = {k.replace(".", "_"): v for k, v in insert_dict[0].items()}
                        text = template_for_tbl[col_name].format(**insert_dict_f_safe)
                        logger.info(f"Text: {text}")
                    elif col.metadata.get("is_time_column", False):
                        # if the col_name is time column, the template is unnecessary
                        logger.info(f"Template for time column {col_name} not found.")
                        self.columns_to_merge_to_text[tbl_name].remove(col_name)
                    else:
                        # if a column is not in the template, remove it from merge_to_text
                        logger.info(f"Template for {col_name} not found.")
                        logger.info(f"Template: {template_for_tbl}")
                        self.columns_to_merge_to_text[tbl_name].remove(col_name)

            if not self.columns_to_merge_to_text[tbl_name] and not with_meaningful_float_columns:
                logger.info(f"No column to merge to text for table {tbl_name}.")
                self.columns_to_merge_to_text[tbl_name] = ["default_value"]
                # generate the default value
                value_dicts = generate_value_dicts(
                    "default_value",
                    DBBColumnDType.text_t,
                    [""],
                    self.modify_function_dict.get(tbl_name, {}),
                )
                value_dicts_f_safe = {k.replace(".", "_"): v for k, v in value_dicts[0].items()}
                text = self.template[tbl_name]["default_value"].format(**value_dicts_f_safe)
                logger.info(f"Text: {text}")

    def transform(self, rdb_data: RDBData, device: DeviceInfo) -> RDBData:
        if True:
            tbl_col_name = [
                (key1, key2)
                for key1 in self.tbl_col_name
                for key2 in self.tbl_col_name[key1]
            ]
            tmp = [self.tbl_col_name[key1][key2] for (key1, key2) in tbl_col_name]
            tmp = self._encode(tmp)
            self.tbl_col_name_emb = {key1: {} for key1 in self.tbl_col_name}
            for (key1, key2), emb in zip(tbl_col_name, tmp):
                self.tbl_col_name_emb[key1][key2] = emb

        for tbl_name, tbl in rdb_data.tables.items():
            for col_name, col in tbl.items():
                col.metadata["name_emb"] = self.tbl_col_name_emb[tbl_name][
                    col_name
                ].tolist()

        for tbl_name, tbl in rdb_data.tables.items():
            # if tbl is empty dict
            if tbl == {}:
                continue
            # For col_name in merge_to_text, we will merge the column to text
            for col_name in self.columns_to_merge_to_text[tbl_name]:
                if col_name == "default_value":
                    # If no column to merge, use the default value
                    value_dicts = [{"default_value": " "}] * self.tbl_data_length[
                        tbl_name
                    ]
                else:
                    col_dtype = tbl[col_name].metadata["dtype"]
                    value_dicts = generate_value_dicts(
                        col_name,
                        col_dtype,
                        tbl[col_name].data,
                        self.modify_function_dict.get(tbl_name, {}),
                    )
                # Generate the text_list and embedding_numpy_list
                # if exists backup, load from backup
                safe_col_name = quote_plus(col_name)
                if not os.path.exists(self.text_backup_dir):
                    os.makedirs(self.text_backup_dir)
                text_backup_path = os.path.join(
                    self.text_backup_dir, f"{tbl_name}_{safe_col_name}.npy"
                )
                if os.path.exists(text_backup_path):
                    text_list = np.load(text_backup_path, allow_pickle=True)
                    text_list = text_list.tolist()
                else:
                    value_dicts_f_safe = [
                        {k.replace(".", "_"): v for k, v in value_dict.items()}
                        for value_dict in value_dicts
                    ]
                    text_list = [
                        self.template[tbl_name][col_name].format(**value_dict_f_safe)
                        for value_dict_f_safe in tqdm(value_dicts_f_safe)
                    ]
                    # Save the text_list to backup, in numpy format
                    text_list = np.array(text_list)
                    np.save(text_backup_path, text_list)

                logger.info(
                    f"Text list sample for {tbl_name} and {col_name}: {text_list[0:5]}"
                )

                embedding_backup_path = os.path.join(
                    self.text_backup_dir, f"{tbl_name}_{safe_col_name}_{self.LLM_name}.npy"
                )
                if os.path.exists(embedding_backup_path):
                    embedding_numpy_list = np.load(embedding_backup_path)
                    logger.info(
                        f"Encoding text for table: {tbl_name} and column: {col_name} done."
                    )
                elif os.path.exists(
                    embedding_backup_path.replace(
                        f"_{self.LLM_name}.npy",
                        f"_{self.LLM_name}_0.npy",
                    )
                ):
                    # If the backup is split into multiple files
                    embedding_numpy_list_concat = []
                    save_idx = 0
                    while os.path.exists(
                        embedding_backup_path.replace(
                            f"_{self.LLM_name}.npy", f"_{self.LLM_name}_{save_idx}.npy"
                        )
                    ):
                        embedding_numpy_list_concat.append(
                            np.load(
                                embedding_backup_path.replace(
                                    f"_{self.LLM_name}.npy",
                                    f"_{self.LLM_name}_{save_idx}.npy",
                                )
                            )
                        )
                        save_idx += 1
                    split_idx = [
                        i * len(text_list) // __SPLIT_NUM__
                        for i in range(__SPLIT_NUM__)
                    ]
                    split_idx.append(len(text_list))
                    logger.info(
                        f"Encoding text for table: {tbl_name} and column: {col_name}"
                    )
                    logger.info(
                        f"Splitting text into {__SPLIT_NUM__} parts. Start from part {save_idx}."
                    )

                    for i in tqdm(range(save_idx, __SPLIT_NUM__)):
                        embedding_numpy_list = self._encode(
                            text_list[split_idx[i] : split_idx[i + 1]],
                            with_dict_accelerate=True,
                        )
                        embedding_numpy_list_concat.append(embedding_numpy_list)
                        np.save(
                            os.path.join(
                                self.text_backup_dir,
                                f"{tbl_name}_{safe_col_name}_{self.LLM_name}_{i}.npy",
                            ),
                            embedding_numpy_list,
                        )

                    logger.info(
                        f"Encoding text for table: {tbl_name} and column: {col_name} done."
                    )
                    embedding_numpy_list = np.concatenate(
                        embedding_numpy_list_concat, axis=0
                    )
                    np.save(embedding_backup_path, embedding_numpy_list)
                elif len(text_list) > __SPLIT_MAX_SIZE__:
                    # split text_list to __SPLIT_NUM__ parts
                    split_idx = [
                        i * len(text_list) // __SPLIT_NUM__
                        for i in range(__SPLIT_NUM__)
                    ]
                    split_idx.append(len(text_list))
                    logger.info(
                        f"Encoding text for table: {tbl_name} and column: {col_name}"
                    )
                    logger.info(f"Splitting text into {__SPLIT_NUM__} parts.")
                    embedding_numpy_list_concat = []
                    for i in tqdm(range(0, __SPLIT_NUM__)):
                        embedding_numpy_list = self._encode(
                            text_list[split_idx[i] : split_idx[i + 1]],
                            with_dict_accelerate=True,
                        )
                        embedding_numpy_list_concat.append(embedding_numpy_list)
                        np.save(
                            os.path.join(
                                self.text_backup_dir,
                                f"{tbl_name}_{safe_col_name}_{self.LLM_name}_{i}.npy",
                            ),
                            embedding_numpy_list,
                        )
                    logger.info(
                        f"Encoding text for table: {tbl_name} and column: {col_name} done."
                    )
                    embedding_numpy_list = np.concatenate(
                        embedding_numpy_list_concat, axis=0
                    )
                    np.save(embedding_backup_path, embedding_numpy_list)
                else:
                    logger.info(
                        f"Encoding text for table: {tbl_name} and column: {col_name}"
                    )
                    embedding_numpy_list = self._encode(
                        text_list,
                        with_dict_accelerate=True,
                    )
                    logger.info(
                        f"Encoding text for table: {tbl_name} and column: {col_name} done."
                    )
                    np.save(embedding_backup_path, embedding_numpy_list)
                combine_embedding = embedding_numpy_list.astype("float32")
                assert combine_embedding.shape == (
                    len(text_list),
                    self.LLM_dim,
                ), f"For table {tbl_name}, Shape mismatch: {combine_embedding.shape} vs {len(text_list), self.LLM_dim}"
                # Add the combined embedding to the table
                metadata_dict = {
                    "name": f"Griffin_text_{col_name}",
                    "dtype": DBBColumnDType.float_t,
                    "name_emb": self.tbl_col_name_emb[tbl_name][col_name].tolist(),
                }
                if is_task_table(tbl_name):
                    metadata_dict["is_target_column"] = False
                tbl[f"Griffin_text_{col_name}"] = ColumnData(
                    data=combine_embedding,
                    metadata=metadata_dict,
                )

            # Only keep the columns in columns_to_keep
            tbl_keys = list(tbl.keys())
            if not self.config.keep_original_columns:
                for col_name in tbl_keys:
                    if col_name not in self.columns_to_keep[
                        tbl_name
                    ] and not col_name.startswith("Griffin_text_"):
                        tbl.pop(col_name)
            else:
                # Only remove the columns representing time but in category dtype
                for col_name in tbl_keys:
                    if (
                        col_name.startswith("YEAR")
                        or col_name.startswith("MONTH")
                        or col_name.startswith("DAY")
                        or col_name.startswith("DAYOFWEEK")
                        # or col_name.startswith("TIMESTAMP")
                        or tbl[col_name].metadata["dtype"] == DBBColumnDType.text_t
                    ):
                        tbl.pop(col_name)
        return rdb_data


def generate_value_dicts(col_name, col_dtype, col_data, modify_function_dict):
    # This function is used to generate the value dicts for the column data
    # The value dict is used for the template to generate the text
    value_dicts = []
    # 1. Check if the col_name is in the modify_function_dict.
    # If yes, apply the modify function to the col_data
    if col_name in modify_function_dict:
        col_data = np.array(
            [eval(modify_function_dict[col_name])(data) for data in col_data]
        )
    # 2. if the col_data is datetime, convert to string
    if col_dtype == DBBColumnDType.timestamp_t:
        # col_data = datetime_utils.dt2ts(col_data)
        day = datetime_utils.dt2day(col_data)
        month = datetime_utils.dt2month(col_data)
        year = datetime_utils.dt2year(col_data)
        dayofweek = datetime_utils.dt2dayofweek(col_data)
        # day and year should be integer.
        day = day.astype(int)
        year = year.astype(int)
        value_dicts = [
            {
                "DAY": day[i],
                "MONTH": MONTH_DICT.get(month[i], month[i]),
                "YEAR": year[i],
                "DAYOFWEEK": DAYOFWEEK_DICT.get(dayofweek[i], dayofweek[i]),
            }
            for i in range(len(col_data))
        ]
        return value_dicts
    # 3. if the col_data is float, round to 2 decimal places
    if isinstance(col_data[0], np.floating):
        col_data = np.round(col_data, 2)
        # if any float is integer, convert it to integer
        col_data_tmp = []
        for i in range(len(col_data)):
            if col_data[i].is_integer():
                # col_data[i] = int(col_data[i])
                col_data_tmp.append(int(col_data[i]))
            else:
                col_data_tmp.append(col_data[i])
        col_data = col_data_tmp

    # 4. if the col_data is too long, cut down it
    col_data = [str(data).strip() for data in col_data]
    col_data = [
        (
            (data[0:500].strip() + "..." + data[-500:].strip())
            if len(data) > 1000
            else data
        )
        for data in col_data
    ]
    col_data = [
        data.replace("\n", " ").replace("\r", " ").replace("  ", " ")
        for data in col_data
    ]
    for i in range(len(col_data)):
        value_dict = {}
        value_dict[col_name] = col_data[i]
        value_dicts.append(value_dict)

    return value_dicts


def generate_target_template(col_name, target_tbl_template):
    if col_name in target_tbl_template:
        return_template = f"Predicting the column {col_name} with: {target_tbl_template[col_name]}"
        return True, return_template
    return False, None


DAYOFWEEK_DICT = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}

MONTH_DICT = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}


TEMPLATE = {
    "purchase": {
        "product": {
            "name": "Feature node of product: The name of the product is {name}.",
            "price": "Feature node of product: The price of the product is {price}.",
            "rating": "Feature node of product: The rating of the product is {rating}.",
            "discount_type": "Feature node of product: The discount type of the product is {discount_type}.",
            "attractiveness": "Feature node of product: The attractiveness of the product is {attractiveness}.",
            "necessity": "Feature node of product: The necessity of the product is {necessity}.",
        },
        "customer": {
            "income": "Feature node of customer: The income of the customer is {income}.",
            "use_frequency": "Feature node of customer: The use frequency of the customer is {use_frequency}.",
            "rating_tendency": "Feature node of customer: The rating tendency of the customer is {rating_tendency}.",
            "platform_engagement": "Feature node of customer: The platform engagement of the customer is {platform_engagement}.",
            "spending_priority": "Feature node of customer: The spending priority of the customer is {spending_priority}.",
        },
        "transaction": {
            "product_name": "Feature node of transaction: The transaction with transaction product {product_name}.",
            "product_rating": "Feature node of transaction: The transaction with rating {product_rating}.",
            "review": "Feature node of transaction: The review is {review}.",
        },
        "__task__:discount:product": {
            "name": "Seed node of product. Predicting the discount type of the product with name {name}.",
            "price": "Seed node of product. Predicting the discount type of the product with price {price}.",
            "rating": "Seed node of product. Predicting the discount type of the product with rating {rating}.",
            "attractiveness": "Seed node of product. Predicting the discount type of the product with attractiveness {attractiveness}.",
            "necessity": "Seed node of product. Predicting the discount type of the product with necessity {necessity}.",
        },
        "__task__:attractiveness:product": {
            "name": "Seed node of product. Predicting the attractiveness of the product {name}.",
            "price": "Seed node of product. Predicting the attractiveness of the product {price}.",
            "rating": "Seed node of product. Predicting the attractiveness of the product {rating}.",
            "discount_type": "Seed node of product. Predicting the attractiveness of the product {discount_type}.",
            "necessity": "Seed node of product. Predicting the attractiveness of the product {necessity}.",
        },
        "__task__:necessity:product": {
            "name": "Seed node of product. Predicting the necessity of the product {name}.",
            "price": "Seed node of product. Predicting the necessity of the product {price}.",
            "rating": "Seed node of product. Predicting the necessity of the product {rating}.",
            "discount_type": "Seed node of product. Predicting the necessity of the product {discount_type}.",
            "attractiveness": "Seed node of product. Predicting the necessity of the product {attractiveness}.",
        },
        "__task__:rating_tendency:customer": {
            "income": "Seed node of customer. Predicting the rating tendency of the customer with income {income}.",
            "use_frequency": "Seed node of customer. Predicting the rating tendency of the customer with use frequency {use_frequency}.",
            "platform_engagement": "Seed node of customer. Predicting the rating tendency of the customer with platform engagement {platform_engagement}.",
            "spending_priority": "Seed node of customer. Predicting the rating tendency of the customer with spending priority {spending_priority}.",
        },
        "__task__:platform_engagement:customer": {
            "income": "Seed node of customer. Predicting the platform engagement of the customer with income {income}.",
            "use_frequency": "Seed node of customer. Predicting the platform engagement of the customer with use frequency {use_frequency}.",
            "rating_tendency": "Seed node of customer. Predicting the platform engagement of the customer with rating tendency {rating_tendency}.",
            "spending_priority": "Seed node of customer. Predicting the platform engagement of the customer with spending priority {spending_priority}.",
        },
        "__task__:spending_priority:customer": {
            "income": "Seed node of customer. Predicting the spending priority of the customer with income {income}.",
            "use_frequency": "Seed node of customer. Predicting the spending priority of the customer with use frequency {use_frequency}.",
            "rating_tendency": "Seed node of customer. Predicting the spending priority of the customer with rating tendency {rating_tendency}.",
            "platform_engagement": "Seed node of customer. Predicting the spending priority of the customer with platform engagement {platform_engagement}.",
        },
    },
    "facebook-recruiting": {
        "Bidders": {
            "payment_account": "Feature node of Bidders: the payment account of the bidder is {payment_account}.",
            "address": "Feature node of Bidders: the address of the bidder is {address}.",
        },
        "Bids": {
            "bidder_id": "Feature node of Bids: the id of the bidder is {bidder_id}.",
            "auction": "Feature node of Bids: the auction of the bid is {auction}.",
            "merchandise": "Feature node of Bids: the merchandise of the bid is {merchandise}.",
            "device": "Feature node of Bids: the device of the bid is {device}.",
            "time": "Feature node of Bids: The bid is on {DAYOFWEEK}, {MONTH} {DAY}, {YEAR}.",
            "country": "Feature node of Bids: the country of the bid is {country}.",
            "ip": "Feature node of Bids: the ip of the bid is {ip}.",
            "url": "Feature node of Bids: the url of the bid is {url}.",
        },
        "__task__:bot:Bidders": {
            "payment_account": "Seed node of Bidders: the payment account of the bidder is {payment_account}.",
            "address": "Seed node of Bidders: the address of the bidder is {address}.",
        },
    },
    "redhat": {
        "People": {
            "char_1": "Feature node of People: one of the characteristic of the people is {char_1}.",
            "group_1": "Feature node of People: the group of the people is {group_1}.",
            "char_2": "Feature node of People: one of the characteristic of the people is {char_2}.",
            "date": "Feature node of People: The bid is on {MONTH} {DAY}, {YEAR}.",
            "char_3": "Feature node of People: one of the characteristic of the people is {char_3}.",
            "char_4": "Feature node of People: one of the characteristic of the people is {char_4}.",
            "char_5": "Feature node of People: one of the characteristic of the people is {char_5}.",
            "char_6": "Feature node of People: one of the characteristic of the people is {char_6}.",
            "char_7": "Feature node of People: one of the characteristic of the people is {char_7}.",
            "char_8": "Feature node of People: one of the characteristic of the people is {char_8}.",
            "char_9": "Feature node of People: one of the characteristic of the people is {char_9}.",
            "char_10": "Feature node of People: one of the characteristic of the people is {char_10}.",
            "char_11": "Feature node of People: one of the characteristic of the people is {char_11}.",
            "char_12": "Feature node of People: one of the characteristic of the people is {char_12}.",
            "char_13": "Feature node of People: one of the characteristic of the people is {char_13}.",
            "char_14": "Feature node of People: one of the characteristic of the people is {char_14}.",
            "char_15": "Feature node of People: one of the characteristic of the people is {char_15}.",
            "char_16": "Feature node of People: one of the characteristic of the people is {char_16}.",
            "char_17": "Feature node of People: one of the characteristic of the people is {char_17}.",
            "char_18": "Feature node of People: one of the characteristic of the people is {char_18}.",
            "char_19": "Feature node of People: one of the characteristic of the people is {char_19}.",
            "char_20": "Feature node of People: one of the characteristic of the people is {char_20}.",
            "char_21": "Feature node of People: one of the characteristic of the people is {char_21}.",
            "char_22": "Feature node of People: one of the characteristic of the people is {char_22}.",
            "char_23": "Feature node of People: one of the characteristic of the people is {char_23}.",
            "char_24": "Feature node of People: one of the characteristic of the people is {char_24}.",
            "char_25": "Feature node of People: one of the characteristic of the people is {char_25}.",
            "char_26": "Feature node of People: one of the characteristic of the people is {char_26}.",
            "char_27": "Feature node of People: one of the characteristic of the people is {char_27}.",
            "char_28": "Feature node of People: one of the characteristic of the people is {char_28}.",
            "char_29": "Feature node of People: one of the characteristic of the people is {char_29}.",
            "char_30": "Feature node of People: one of the characteristic of the people is {char_30}.",
            "char_31": "Feature node of People: one of the characteristic of the people is {char_31}.",
            "char_32": "Feature node of People: one of the characteristic of the people is {char_32}.",
            "char_33": "Feature node of People: one of the characteristic of the people is {char_33}.",
            "char_34": "Feature node of People: one of the characteristic of the people is {char_34}.",
            "char_35": "Feature node of People: one of the characteristic of the people is {char_35}.",
            "char_36": "Feature node of People: one of the characteristic of the people is {char_36}.",
            "char_37": "Feature node of People: one of the characteristic of the people is {char_37}.",
            "char_38": "Feature node of People: one of the characteristic of the people is {char_38}.",
        },
        "Action": {
            "date": "Feature node of Action: The action is on {MONTH} {DAY}, {YEAR}.",
            "activity_category": "Feature node of Action: the category of the activity is {activity_category}.",
            "char_1": "Feature node of Action: one of the characteristic of the action is {char_1}.",
            "char_2": "Feature node of Action: one of the characteristic of the action is {char_2}.",
            "char_3": "Feature node of Action: one of the characteristic of the action is {char_3}.",
            "char_4": "Feature node of Action: one of the characteristic of the action is {char_4}.",
            "char_5": "Feature node of Action: one of the characteristic of the action is {char_5}.",
            "char_6": "Feature node of Action: one of the characteristic of the action is {char_6}.",
            "char_7": "Feature node of Action: one of the characteristic of the action is {char_7}.",
            "char_8": "Feature node of Action: one of the characteristic of the action is {char_8}.",
            "char_9": "Feature node of Action: one of the characteristic of the action is {char_9}.",
            "char_10": "Feature node of Action: one of the characteristic of the action is {char_10}.",
            "outcome": "Feature node of Action: the outcome of the action is {outcome}.",
        },
        "__task__:bval_pred:Action": {
            "date": "Seed node of Action: The action is on {MONTH} {DAY}, {YEAR}.",
            "activity_category": "Seed node of Action: the category of the activity is {activity_category}.",
            "char_1": "Seed node of Action: one of the characteristic of the action is {char_1}.",
            "char_2": "Seed node of Action: one of the characteristic of the action is {char_2}.",
            "char_3": "Seed node of Action: one of the characteristic of the action is {char_3}.",
            "char_4": "Seed node of Action: one of the characteristic of the action is {char_4}.",
            "char_5": "Seed node of Action: one of the characteristic of the action is {char_5}.",
            "char_6": "Seed node of Action: one of the characteristic of the action is {char_6}.",
            "char_7": "Seed node of Action: one of the characteristic of the action is {char_7}.",
            "char_8": "Seed node of Action: one of the characteristic of the action is {char_8}.",
            "char_9": "Seed node of Action: one of the characteristic of the action is {char_9}.",
            "char_10": "Seed node of Action: one of the characteristic of the action is {char_10}.",
            "outcome": "Seed node of Action: the outcome of the action is {outcome}.",
        },
    },
    "talkingdata": {
        "Gender_age": {
            "default_value": "Feature node of Gender_age.",
        },
        "Brand": {
            "phone_brand": "Feature node of Brand: the brand of the phone is {phone_brand}.",
            "device_model": "Feature node of Brand: the device model of the phone is {device_model}.",
        },
        "App_labels": {
            "default_value": "Feature node of App_labels.",
        },
        "App_events": {
            "is_active": "Feature node of App_events: whether the app is in active use when the event occurs is {is_active}.",
        },
        "Label_categories": {
            "category": "Feature node of Label_categories:the category of the app in text is {category}.",
        },
        "Events": {
            "timestamp": "Feature node of Events: on {DAYOFWEEK}, {MONTH} {DAY}, {YEAR}.",
            "longitude": "Feature node of Events: the longitude of the event is {longitude}.",
            "latitude": "Feature node of Events: the latitude of the event is {latitude}.",
        },
        "Apps": {
            "default_value": "Feature node of Apps.",
        },
        "__task__:demo-pred:Gender_age": {
            "default_value": "Seed node of Gender_age.",
        },
    },
    "elo-merchant-recommendation": {
        "Card": {
            "first_active_month": "Feature node of Card: the date when the card started to be active is {MONTH} {DAY}, {YEAR}.",
            "feature_1": "Feature node of Card: one of the feature is {feature_1}.",
            "feature_2": "Feature node of Card: one of the feature is {feature_2}.",
            "feature_3": "Feature node of Card: one of the feature is {feature_3}.",
        },
        "historical_transactions": {
            "authorized_flag": "Feature node of historical_transactions: whether the transaction is authorized is {authorized_flag}.",
            "category_1": "Feature node of historical_transactions: one of the categories of the transaction is {category_1}.",
            "installments": "Feature node of historical_transactions: the installments of the transaction is {installments}.",
            "category_3": "Feature node of historical_transactions: one of the categories of the transaction is {category_3}.",
            "merchant_category_id": "Feature node of historical_transactions: the category id of the merchant is {merchant_category_id}.",
            "month_lag": "Feature node of historical_transactions: the number of months between the transaction date and the reference date is {month_lag}.",
            "purchase_amount": "Feature node of historical_transactions: the amount of the purchase is {purchase_amount}.",
            "purchase_date": "Feature node of historical_transactions: the time of the purchase is {DAYOFWEEK}, {MONTH} {DAY}, {YEAR}.",
            "category_2": "Feature node of historical_transactions: one of the categories of the transaction is {category_2}.",
        },
        "new_merchant_transactions": {
            "category_1": "Feature node of new_merchant_transactions: one of the categories of the transaction is {category_1}.",
            "installments": "Feature node of new_merchant_transactions: the installments of the transaction is {installments}.",
            "category_3": "Feature node of new_merchant_transactions: one of the categories of the transaction is {category_3}.",
            "merchant_category_id": "Feature node of new_merchant_transactions: the category id of the merchant is {merchant_category_id}.",
            "month_lag": "Feature node of new_merchant_transactions: the number of months between the transaction date and the reference date is {month_lag}.",
            "purchase_amount": "Feature node of new_merchant_transactions: the amount of the purchase is {purchase_amount}.",
            "purchase_date": "Feature node of new_merchant_transactions: the time of the purchase is {DAYOFWEEK}, {MONTH} {DAY}, {YEAR}.",
            "category_2": "Feature node of new_merchant_transactions: one of the categories of the transaction is {category_2}.",
            "category_1": "Feature node of new_merchant_transactions: one of the categories of the transaction is {category_1}.",
        },
        "Merchants": {
            "merchant_group_id": "Feature node of Merchants: the merchant group id is {merchant_group_id}.",
            "merchant_category_id": "Feature node of Merchants: the merchant category id is {merchant_category_id}.",
            "numerical_1": "Feature node of Merchants: one of the numerical feature is {numerical_1}.",
            "numerical_2": "Feature node of Merchants: one of the numerical feature is {numerical_2}.",
            "category_1": "Feature node of Merchants: one of the categorical feature is {category_1}.",
            "most_recent_sales_range": "Feature node of Merchants: the range of the most recent sales is {most_recent_sales_range}.",
            "most_recent_purchases_range": "Feature node of Merchants: the range of the most recent purchases is {most_recent_purchases_range}.",
            "avg_sales_lag3": "Feature node of Merchants: the average sales of the past three months is {avg_sales_lag3}.",
            "avg_purchases_lag3": "Feature node of Merchants: the average purchases of the past three months is {avg_purchases_lag3}.",
            "active_months_lag3": "Feature node of Merchants: the number of active months of the past three months is {active_months_lag3}.",
            "avg_sales_lag6": "Feature node of Merchants: the average sales of the past six months is {avg_sales_lag6}.",
            "avg_purchases_lag6": "Feature node of Merchants: the average purchases of the past six months is {avg_purchases_lag6}.",
            "active_months_lag6": "Feature node of Merchants: the number of active months of the past six months is {active_months_lag6}.",
            "avg_sales_lag12": "Feature node of Merchants: the average sales of the past twelve months is {avg_sales_lag12}.",
            "avg_purchases_lag12": "Feature node of Merchants: the average purchases of the past twelve months is {avg_purchases_lag12}.",
            "active_months_lag12": "Feature node of Merchants: the number of active months of the past twelve months is {active_months_lag12}.",
            "category_4": "Feature node of Merchants: one of the categorical feature is {category_4}.",
            "category_2": "Feature node of Merchants: one of the categorical feature is {category_2}.",
        },
        "City": {
            "default_value": "Feature node of City.",
        },
        "State": {
            "default_value": "Feature node of State.",
        },
        "Subsector": {
            "default_value": "Feature node of Subsector.",
        },
        "__task__:recommendation:Card": {
            "first_active_month": "Seed node of Card: the date when the card started to be active is {MONTH} {DAY}, {YEAR}.",
            "feature_1": "Seed node of Card: one of the feature is {feature_1}.",
            "feature_2": "Seed node of Card: one of the feature is {feature_2}.",
            "feature_3": "Seed node of Card: one of the feature is {feature_3}.",
        },
    },
    "airbnb": {
        "User": {
            "date_account_created": "Feature node of User: the time when the account was created is {MONTH} {DAY}, {YEAR}.",
            "timestamp_first_active": "Feature node of User: the time when user was active for the first time is {DAYOFWEEK}, {MONTH} {DAY}, {YEAR}.",
            "date_first_booking": "Feature node of User: the date when the user made the first booking is {MONTH} {DAY}, {YEAR}.",
            "age": "Feature node of User: the age of the user is {age}.",
            "signup_method": "Feature node of User: the method how the user signed up is {signup_method}.",
            "signup_flow": "Feature node of User: the signup flow of the user is {signup_flow}.",
            "language": "Feature node of User: the language that the user chose is {language}.",
            "affiliate_channel": "Feature node of User: the marketing channel through which the user was referred to the platform is {affiliate_channel}.",
            "affiliate_provider": "Feature node of User: the provider of the marketing is {affiliate_provider}.",
            "first_affiliate_tracked": "Feature node of User: the first marketing the user interacted with before the signing up is {first_affiliate_tracked}.",
            "signup_app": "Feature node of User: the app through which the user signed up is {signup_app}.",
            "first_device_type": "Feature node of User: the type of the device used when the user visited the platform for the first time is {first_device_type}.",
            "first_browser": "Feature node of User: the browser used when the user visited the platform for the first time is {first_browser}.",
        },
        "Population": {
            "population_in_thousands": "Feature node of Population: the population in thousands of the corresponding bucket is {population_in_thousands}.",
            "year": "Feature node of Population: the year is {YEAR}.",
        },
        "Country": {
            "lat_destination": "Feature node of Country: the latitude of the destination country is {lat_destination}.",
            "lng_destination": "Feature node of Country: the longitude of the destination country is {lng_destination}.",
            "distance_km": "Feature node of Country: the distance between the place of the user and the destination country is {distance_km}.",
            "destination_km2": "Feature node of Country: the territorial size of country in square kilometers is {destination_km2}.",
            "destination_language": "Feature node of Country: the language of the destination country is {destination_language}.",
            "language_levenshtein_distance": "Feature node of Country: the levenshtein distance between the user's language and the language of the destination country is {language_levenshtein_distance}.",
        },
        "Session": {
            "action": "Feature node of Session: the action of the user in this session is {action}.",
            "action_type": "Feature node of Session: the type of the action of the user in this session is {action_type}.",
            "action_detail": "Feature node of Session: the action detail of the user in this session is {action_detail}.",
            "device_type": "Feature node of Session: the type of the device that the user used in this session is {device_type}.",
            "secs_elapsed": "Feature node of Session: the amount of time in seconds that the user spent on the action during the session is {secs_elapsed}.",
        },
        "Gender": {
            "default_value": "Feature node of Gender.",
        },
        "Age_bucket": {
            "default_value": "Feature node of Age_bucket.",
        },
        "__task__:destination:User": {
            "date_account_created": "Seed node of User: the time when the account was created is {MONTH} {DAY}, {YEAR}.",
            "timestamp_first_active": "Seed node of User: the time when user was active for the first time is {DAYOFWEEK}, {MONTH} {DAY}, {YEAR}.",
            "date_first_booking": "Seed node of User: the date when the user made the first booking is {MONTH} {DAY}, {YEAR}.",
            "age": "Seed node of User: the age of the user is {age}.",
            "signup_method": "Seed node of User: the method how the user signed up is {signup_method}.",
            "signup_flow": "Seed node of User: the signup flow of the user is {signup_flow}.",
            "language": "Seed node of User: the language that the user chose is {language}.",
            "affiliate_channel": "Seed node of User: the marketing channel through which the user was referred to the platform is {affiliate_channel}.",
            "affiliate_provider": "Seed node of User: the provider of the marketing is {affiliate_provider}.",
            "first_affiliate_tracked": "Seed node of User: the first marketing the user interacted with before the signing up is {first_affiliate_tracked}.",
            "signup_app": "Seed node of User: the app through which the user signed up is {signup_app}.",
            "first_device_type": "Seed node of User: the type of the device used when the user visited the platform for the first time is {first_device_type}.",
            "first_browser": "Seed node of User: the browser used when the user visited the platform for the first time is {first_browser}.",
        },
    },
}
