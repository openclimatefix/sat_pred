"""Tests for opening and lazily concatenating tensorstore-backed zarr stores"""

import numpy as np
import pytest
import xarray as xr

from sat_pred.dataset import open_sat_data
from sat_pred.xr_tensorstore import concat_tensorstore, open_zarr_paths
from tests.conftest import CHANNEL_NAMES, IMAGE_SIZE, NUM_CHANNELS


def test_open_single_store(sat_zarr_path):
    """A single store is opened without any concatenation"""
    ds = open_zarr_paths(sat_zarr_path)

    assert set(ds.sizes) == {"time", "channel", "y_geostationary", "x_geostationary"}
    assert ds.sizes["channel"] == NUM_CHANNELS


def test_open_a_list_of_one_store(sat_zarr_path):
    """A one-element list gives the same result as the bare path"""
    ds_from_list = open_zarr_paths([sat_zarr_path], concat_dim="time")
    ds_from_str = open_zarr_paths(sat_zarr_path)

    assert ds_from_list.equals(ds_from_str)


def test_open_multiple_stores(sat_zarr_path, sat_zarr_paths_split):
    """Stores split by time join back into the data they were split from"""
    ds_split = open_zarr_paths(sat_zarr_paths_split, concat_dim="time")
    ds_whole = open_zarr_paths(sat_zarr_path)

    assert ds_split.sizes == ds_whole.sizes
    np.testing.assert_array_equal(ds_split.time.values, ds_whole.time.values)
    np.testing.assert_array_equal(ds_split["data"].values, ds_whole["data"].values)

    # The attributes of the first store are kept
    assert ds_split.attrs == ds_whole.attrs


def test_open_multiple_stores_stays_lazy(sat_zarr_paths_split):
    """Concatenating does not pull the data into memory"""
    ds = open_zarr_paths(sat_zarr_paths_split, concat_dim="time")

    assert not isinstance(ds["data"].variable._data, np.ndarray)


def test_open_glob(sat_zarr_paths_split):
    """A glob pattern picks up all the matching stores"""
    directory = sat_zarr_paths_split[0].rsplit("/", 1)[0]

    ds_glob = open_zarr_paths(f"{directory}/*.zarr", concat_dim="time")
    ds_list = open_zarr_paths(sorted(sat_zarr_paths_split), concat_dim="time")

    np.testing.assert_array_equal(ds_glob.time.values, ds_list.time.values)


def test_open_multiple_stores_needs_a_concat_dim(sat_zarr_paths_split):
    """Opening several stores without saying how to join them is an error"""
    with pytest.raises(ValueError, match="concat_dim"):
        open_zarr_paths(sat_zarr_paths_split)


def test_open_no_stores(tmp_path):
    """A glob which matches nothing is an error rather than an empty dataset"""
    with pytest.raises(ValueError, match="No Zarr stores found"):
        open_zarr_paths(f"{tmp_path}/*.zarr", concat_dim="time")


def test_concat_rejects_mismatched_stores(tmp_path, sat_zarr_path, sat_datetimes):
    """Stores which do not line up outside the concat dimension are rejected"""
    rng = np.random.default_rng(seed=2)

    # A store with a different image size
    ds_odd = xr.Dataset(
        data_vars={
            "data": (
                ["time", "channel", "y_geostationary", "x_geostationary"],
                rng.random((4, NUM_CHANNELS, IMAGE_SIZE // 2, IMAGE_SIZE), dtype=np.float32),
            )
        },
        coords={
            "time": sat_datetimes[:4],
            "channel": CHANNEL_NAMES,
            "y_geostationary": np.arange(IMAGE_SIZE // 2, dtype=np.float64) * 3000,
            "x_geostationary": np.arange(IMAGE_SIZE, dtype=np.float64) * -3000,
        },
    )
    odd_path = f"{tmp_path}/odd.zarr"
    ds_odd.to_zarr(odd_path, zarr_format=3)

    with pytest.raises(ValueError, match="y_geostationary"):
        open_zarr_paths([sat_zarr_path, odd_path], concat_dim="time")


def test_concat_rejects_a_missing_concat_dim(sat_zarr_paths_split):
    """Concatenating along a dimension the stores do not have is an error"""
    datasets = [open_zarr_paths(path) for path in sat_zarr_paths_split]

    with pytest.raises(ValueError, match="not a dimension"):
        concat_tensorstore(datasets, concat_dim="not_a_dim")


def test_open_sat_data_from_split_stores(sat_zarr_path, sat_zarr_paths_split):
    """The satellite loader gives the same data whether it is split across stores or not"""
    da_split = open_sat_data(sat_zarr_paths_split)
    da_whole = open_sat_data(sat_zarr_path)

    assert "time_utc" in da_split.dims
    assert da_split.sizes == da_whole.sizes
    np.testing.assert_array_equal(da_split.time_utc.values, da_whole.time_utc.values)
    np.testing.assert_array_equal(da_split.values, da_whole.values)
