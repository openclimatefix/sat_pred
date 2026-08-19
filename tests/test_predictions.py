"""Tests for the shared layout of the prediction stores"""

import numpy as np
import pandas as pd
import xarray as xr

from sat_pred.dataset import open_sat_data
from sat_pred.predictions import PREDICTION_DIMS, prediction_coords
from tests.conftest import (
    CHANNEL_NAMES,
    DATA_FREQ_MINS,
    FORECAST_MINS,
    IMAGE_SIZE,
    NUM_FORECAST_STEPS,
)


def test_prediction_coords_are_in_prediction_dims_order(sat_zarr_path):
    """The whole point of the module: both writers lay a store out the same way

    The coords are built as a dict and then used as the dim order, so a store written from them is
    only laid out right if the keys come back in `PREDICTION_DIMS` order.
    """
    da = open_sat_data(sat_zarr_path)

    coords = prediction_coords(da, forecast_mins=FORECAST_MINS, sample_freq_mins=DATA_FREQ_MINS)

    assert ("init_time_utc", *coords) == PREDICTION_DIMS


def test_prediction_coords_come_off_the_input_data(sat_zarr_path):
    """The predictions cover the same channels and the same grid as the data they were made from"""
    da = open_sat_data(sat_zarr_path)

    coords = prediction_coords(da, forecast_mins=FORECAST_MINS, sample_freq_mins=DATA_FREQ_MINS)

    assert list(coords["channel"]) == CHANNEL_NAMES
    np.testing.assert_array_equal(coords["y_geostationary"], da.y_geostationary.values)
    np.testing.assert_array_equal(coords["x_geostationary"], da.x_geostationary.values)


def test_a_store_built_from_the_coords_has_the_expected_shape(sat_zarr_path):
    """Both writers build their array as `{"init_time_utc": ..., **prediction_coords(...)}`"""
    da = open_sat_data(sat_zarr_path)
    init_times = pd.to_datetime(["2020-01-01 12:00", "2020-01-01 12:30"])

    coords = prediction_coords(da, forecast_mins=FORECAST_MINS, sample_freq_mins=DATA_FREQ_MINS)
    shape = (len(init_times), *(len(values) for values in coords.values()))

    da_y_hat = xr.DataArray(
        np.zeros(shape, dtype=np.float16),
        dims=PREDICTION_DIMS,
        coords={"init_time_utc": init_times, **coords},
    )

    assert da_y_hat.sizes == {
        "init_time_utc": len(init_times),
        "channel": len(CHANNEL_NAMES),
        "step": NUM_FORECAST_STEPS,
        "y_geostationary": IMAGE_SIZE,
        "x_geostationary": IMAGE_SIZE,
    }
