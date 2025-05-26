import copy
from typing import Tuple, Dict, Optional, List
import pydantic
import numpy as np
import pandas as pd
import logging
from dbinfer_bench import DBBColumnDType

from ...device import DeviceInfo
from .base import (
    ColumnTransform,
    column_transform,
    ColumnData,
    RDBData,
)

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

class RemapCategoryTransformConfig(pydantic.BaseModel):
    with_category_mapping : bool = False

@column_transform
class RemapCategoryTransform(ColumnTransform):
    config_class = RemapCategoryTransformConfig
    name = "remap_category"
    input_dtype = DBBColumnDType.category_t
    output_dtypes = [DBBColumnDType.category_t]
    output_name_formatters : List[str] = ["{name}"]

    def __init__(self, config : RemapCategoryTransformConfig):
        super().__init__(config)

    def fit(
        self,
        column : ColumnData,
        device : DeviceInfo
    ) -> None:
        if column.data.ndim > 1:
            raise ValueError("RemapCategoryTransform only supports 1D data.")
        _, self.categories = pd.factorize(column.data, sort=True, use_na_sentinel=True)
        self.unseen_category = len(self.categories)
        # output the reverse dictionary of category mapping
        self.category_mapping = dict(zip(range(len(self.categories)), self.categories))
        logger.info(f"Category mapping: {self.category_mapping}")

    def transform(
        self,
        column : ColumnData,
        device : DeviceInfo
    ) -> List[ColumnData]:
        if column.data.ndim > 1:
            raise ValueError("RemapCategoryTransform only supports 1D data.")

        new_data = pd.Categorical(column.data, categories=self.categories).codes.copy()
        new_data[new_data == -1] = self.unseen_category
        new_data = new_data.astype('int64')
        new_meta = copy.deepcopy(column.metadata)
        new_meta['num_categories'] = len(self.categories) + 1
        # Save the mapping to new_meta
        if self.config.with_category_mapping:
            self.category_mapping = {int(k): str(v) for k, v in self.category_mapping.items()}
            new_meta['category_mapping'] = self.category_mapping

        return [ColumnData(new_meta, new_data)]


class CategoryFrequencyTransformConfig(pydantic.BaseModel):
    pass


@column_transform
class CategoryFrequencyTransform(ColumnTransform):
    config_class = CategoryFrequencyTransformConfig
    name = "category_frequency"
    input_dtype = DBBColumnDType.category_t

    def __init__(self, config: CategoryFrequencyTransformConfig):
        super().__init__(config)

    def fit(self, column: ColumnData, device: DeviceInfo) -> None:
        if (
            column.metadata["name"].startswith("YEAR")
            or column.metadata["name"].startswith("MONTH")
            or column.metadata["name"].startswith("DAY")
            or column.metadata["name"].startswith("DAYOFWEEK")
        ):
            return
        self.freq = pd.Series(column.data).value_counts(normalize=True, dropna=False).to_dict()

    def transform(self, column: ColumnData, device: DeviceInfo) -> List[ColumnData]:
        if (
            column.metadata["name"].startswith("YEAR")
            or column.metadata["name"].startswith("MONTH")
            or column.metadata["name"].startswith("DAY")
            or column.metadata["name"].startswith("DAYOFWEEK")
        ):
            self.output_dtypes = [DBBColumnDType.category_t]
            self.output_name_formatters = ["{name}"]
            return [column]

        new_data = 1 - pd.Series(column.data).map(self.freq).fillna(0).astype("float32").values
        new_meta = copy.deepcopy(column.metadata)
        self.output_dtypes = [DBBColumnDType.category_t, DBBColumnDType.float_t]
        self.output_name_formatters = ["{name}", "Reversed_Freq({name})"]
        new_meta.pop("name")

        return [ColumnData(new_meta, column.data), ColumnData(new_meta, new_data)]
