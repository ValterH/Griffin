import copy
from typing import Tuple, Dict, Optional, List
import pydantic
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer
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
logger.setLevel("DEBUG")


class DatetimeFrequencyTransformConfig(pydantic.BaseModel):
    impute_strategy: Optional[str] = "median"
    bin_method: Optional[str] = "freedman"


@column_transform
class DatetimeFrequencyTransform(ColumnTransform):
    config_class = DatetimeFrequencyTransformConfig
    name = "datetime_frequency"
    input_dtype = DBBColumnDType.timestamp_t
    output_dtypes = [DBBColumnDType.timestamp_t, DBBColumnDType.float_t]
    output_name_formatters: List[str] = ["{name}", "Reversed_Freq({name})"]

    def __init__(self, config: DatetimeFrequencyTransformConfig):
        super().__init__(config)

    def fit(self, column: ColumnData, device: DeviceInfo) -> None:
        self.new_meta = {
            "dtype": self.output_dtypes[0],
            "in_size": 1 if column.data.ndim == 1 else column.data.shape[1],
        }

        if column.data.ndim > 1:
            # Ignore vector embeddings.
            return

        self.transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=self.config.impute_strategy)),
                ("scaler", StandardScaler()),
            ]
        )
        self.transformer.fit(column.data.reshape(-1, 1))
        tmp_data = self.transformer.transform(column.data.reshape(-1, 1)).reshape(
            -1
        )
        self.bins = compute_bins(tmp_data, self.config.bin_method)

    def transform(self, column: ColumnData, device: DeviceInfo) -> List[ColumnData]:
        if column.data.ndim > 1:
            new_data = column.data.astype("float32")
        else:
            if not self.config.frequency_mode:
                new_data = self.transformer.transform(
                    column.data.reshape(-1, 1)
                ).reshape(-1)
                new_data = new_data.astype("float32")
                return [ColumnData(self.new_meta, new_data)]
            else:
                new_data = (
                    self.transformer.transform(column.data.reshape(-1, 1)).reshape(-1)
                ).astype("float32")
                reversed_frequencies, _ = compute_reversed_frequencies(
                    new_data, self.bins
                )
                output_dtype = DBBColumnDType.float_t
                self.output_dtypes = [output_dtype, output_dtype]
                if column.metadata["name"].startswith("TIMESTAMP"):
                    self.output_name_formatters = [
                        "{name}",
                        ("Reversed_Freq({name})", "10:-1"),
                    ]
                else:
                    self.output_name_formatters = ["{name}", "Reversed_Freq({name})"]
                return [
                    ColumnData(self.new_meta, new_data),
                    ColumnData(self.new_meta, reversed_frequencies),
                ]


def freedman_diaconis_bin_width(data):
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    n = len(data)
    bin_width = 2 * iqr / np.cbrt(n)
    return bin_width


def compute_bins(data, bin_method="freedman"):
    if bin_method == "freedman":
        bin_width = freedman_diaconis_bin_width(data)
    else:
        raise NotImplementedError(f"Unknown bin method: {bin_method}")
    bin_edges = np.arange(data.min(), data.max() + bin_width, bin_width)
    return bin_edges


def compute_reversed_frequencies(data, bin_edges):
    # check if data is in bin_edges ranges
    if np.any(data < bin_edges[0]) or np.any(data > bin_edges[-1]):
        # set data to the closest bin edge
        data = np.clip(data, bin_edges[0], bin_edges[-1])

    hist = np.histogram(data, bins=bin_edges)[0]

    # Map each data point to its frequency
    bin_indices = np.digitize(data, bin_edges) - 1
    frequencies = hist[bin_indices]
    frequencies = 1 - (frequencies / len(data))

    return frequencies, bin_edges
