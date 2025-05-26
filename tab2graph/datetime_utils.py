"""Datetime utilities."""
import pandas as pd
import dask.dataframe as dd
import numpy as np

def dask_dt2ts(dt : dd.Series) -> dd.Series:
    """Convert a Dask datetime series to epoch timestamp series."""
    return dt.map_partitions(lambda x: dt2ts(x))

def dt2ts(dt : np.ndarray) -> np.ndarray:
    dt = dt.astype('datetime64[ns]')
    return (dt - np.array(0).astype('datetime64[ns]')).astype('int64')

def ts2dt(ts : np.ndarray) -> np.ndarray:
    return np.array(ts).astype('datetime64[ns]')

def dt2year(dt : np.ndarray) -> np.ndarray:
    dt_series = pd.to_datetime(pd.Series(dt))
    return dt_series.dt.year.values

def dt2month(dt : np.ndarray) -> np.ndarray:
    dt_series = pd.to_datetime(pd.Series(dt))
    return dt_series.dt.month.values

def dt2day(dt : np.ndarray) -> np.ndarray:
    dt_series = pd.to_datetime(pd.Series(dt))
    return dt_series.dt.day.values

def dt2dayofweek(dt : np.ndarray) -> np.ndarray:
    dt_series = pd.to_datetime(pd.Series(dt))
    return dt_series.dt.dayofweek.values
