"""Fixtures shared between the tests

The satellite fixtures build a small synthetic zarr store in the same layout as the training data,
so the tests can exercise the real data loading path without needing the real satellite archive.
"""

import json
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
import yaml

from sat_pred.channels import ChannelConfig
from sat_pred.constants import DATA_CONFIG_NAME, FULL_CONFIG_NAME, MODEL_CONFIG_NAME

# The channel config the training configs use
CHANNELS_PATH = Path(__file__).parents[1] / "configs/datamodule/channels/seviri_rss.yaml"

# The synthetic satellite data covers this period at this frequency...
DATA_START = "2020-01-01 00:00"
DATA_END = "2020-01-06 00:00"
DATA_FREQ_MINS = 15

# ...with this chunk of data missing, so the tests can check that samples are never taken across
# a gap in the data
GAP_START = "2020-01-04 00:00"
GAP_END = "2020-01-04 12:00"

NUM_CHANNELS = 11
IMAGE_SIZE = 32

# Samples are 4 frames of history (including t0) and 4 frames of target
HISTORY_MINS = 45
FORECAST_MINS = 60
NUM_HISTORY_STEPS = HISTORY_MINS // DATA_FREQ_MINS + 1
NUM_FORECAST_STEPS = FORECAST_MINS // DATA_FREQ_MINS

CHANNEL_NAMES = [
    "IR_016", "IR_039", "IR_087", "IR_097", "IR_108", "IR_120",
    "IR_134", "VIS006", "VIS008", "WV_062", "WV_073",
]

AREA_STRING = json.dumps(
    {
        "msg_seviri_rss_3km": {
            "description": "Test area definition",
            "projection": {"proj": "geos", "lon_0": 9.5, "h": 35785831},
        }
    }
)


@pytest.fixture(scope="session")
def channel_config() -> ChannelConfig:
    """The channel config the training configs use"""
    return ChannelConfig.from_yaml(CHANNELS_PATH)


def synthetic_sat_values(num_times: int, seed: int) -> np.ndarray:
    """Random satellite data in the physical units and ranges of each channel

    The channels cover very different ranges - reflectance percentages for the visible channels,
    brightness temperatures in Kelvin for the rest - so the tests use plausible values rather than
    the same range for every channel, otherwise normalisation would clip most of them flat.
    """
    rng = np.random.default_rng(seed)
    constants = ChannelConfig.from_yaml(CHANNELS_PATH)

    data = np.empty((num_times, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    for i, channel in enumerate(CHANNEL_NAMES):
        values = constants[channel]
        data[:, i] = rng.uniform(
            values.clip_min, values.clip_max, size=(num_times, IMAGE_SIZE, IMAGE_SIZE)
        )
    return data


@pytest.fixture(scope="session")
def sat_datetimes() -> pd.DatetimeIndex:
    """The timestamps present in the synthetic satellite data"""
    datetimes = pd.date_range(DATA_START, DATA_END, freq=f"{DATA_FREQ_MINS}min")
    return datetimes[(datetimes < GAP_START) | (datetimes > GAP_END)]


@pytest.fixture(scope="session")
def sat_zarr_path(tmp_path_factory, sat_datetimes) -> str:
    """Path to a small synthetic satellite zarr store"""
    ds = xr.Dataset(
        data_vars={
            "data": (
                ["time", "channel", "y_geostationary", "x_geostationary"],
                synthetic_sat_values(len(sat_datetimes), seed=0),
            )
        },
        coords={
            "time": sat_datetimes,
            "channel": CHANNEL_NAMES,
            "y_geostationary": np.arange(IMAGE_SIZE, dtype=np.float64) * 3000,
            "x_geostationary": np.arange(IMAGE_SIZE, dtype=np.float64) * -3000,
        },
        attrs={"area": AREA_STRING},
    )

    zarr_path = f"{tmp_path_factory.mktemp('sat_data')}/sat.zarr"
    ds.to_zarr(zarr_path, zarr_format=3)

    return zarr_path


@pytest.fixture(scope="session")
def sat_zarr_paths_split(tmp_path_factory, sat_datetimes) -> list[str]:
    """The same data as `sat_zarr_path`, split by time into two stores

    The real training data is stored as one zarr per year, so this is the layout the training
    configs use.
    """
    data = synthetic_sat_values(len(sat_datetimes), seed=0)

    ds = xr.Dataset(
        data_vars={"data": (["time", "channel", "y_geostationary", "x_geostationary"], data)},
        coords={
            "time": sat_datetimes,
            "channel": CHANNEL_NAMES,
            "y_geostationary": np.arange(IMAGE_SIZE, dtype=np.float64) * 3000,
            "x_geostationary": np.arange(IMAGE_SIZE, dtype=np.float64) * -3000,
        },
        attrs={"area": AREA_STRING},
    )

    tmp_path = tmp_path_factory.mktemp("sat_data_split")

    split_at = "2020-01-03 00:00"
    zarr_paths = []
    for name, ds_part in [
        ("first.zarr", ds.sel(time=slice(None, split_at))),
        ("second.zarr", ds.sel(time=slice(split_at, None)).isel(time=slice(1, None))),
    ]:
        zarr_path = f"{tmp_path}/{name}"
        ds_part.to_zarr(zarr_path, zarr_format=3)
        zarr_paths.append(zarr_path)

    return zarr_paths


@pytest.fixture(scope="session")
def sat_zarr_path_with_nans(tmp_path_factory, sat_datetimes) -> str:
    """Path to a synthetic satellite zarr store where one timestamp is entirely NaN"""
    data = synthetic_sat_values(len(sat_datetimes), seed=1)
    data[10] = np.nan

    ds = xr.Dataset(
        data_vars={
            "data": (["time", "channel", "y_geostationary", "x_geostationary"], data),
        },
        coords={
            "time": sat_datetimes,
            "channel": CHANNEL_NAMES,
            "y_geostationary": np.arange(IMAGE_SIZE, dtype=np.float64) * 3000,
            "x_geostationary": np.arange(IMAGE_SIZE, dtype=np.float64) * -3000,
        },
        attrs={"area": AREA_STRING},
    )

    zarr_path = f"{tmp_path_factory.mktemp('sat_data_nans')}/sat.zarr"
    ds.to_zarr(zarr_path, zarr_format=3)

    return zarr_path


@pytest.fixture(scope="session")
def tiny_model_config() -> dict:
    """Config for a SimVP small enough to train in a test"""
    return {
        "_target_": "sat_pred.models.simvp_model.SimVP",
        "num_channels": NUM_CHANNELS,
        "history_len": NUM_HISTORY_STEPS,
        "forecast_len": NUM_FORECAST_STEPS,
        "spatial_size": [IMAGE_SIZE, IMAGE_SIZE],
        "hid_S": 4,
        "hid_T": 8,
        "N_S": 2,
        # SimVP's translator needs at least two layers - it has a skip connection between them
        "N_T": 2,
        "incep_ker": [3],
        "groups": 1,
    }


@pytest.fixture(scope="session")
def training_module_config(tiny_model_config) -> dict:
    """Config for a training module wrapping the tiny model"""
    return {
        "_target_": "sat_pred.training_module.TrainingModule",
        "model": tiny_model_config,
        "optimizer": {"_target_": "sat_pred.optimizers.AdamW", "lr": 0.001},
        "target_loss": "MAE",
    }


@pytest.fixture
def checkpoint_dir(tmp_path, training_module_config) -> str:
    """A checkpoint directory in the layout that training saves

    Holds the model config, the data config, the full experiment config, and a checkpoint file.
    """
    training_module = hydra.utils.instantiate(training_module_config)

    (tmp_path / MODEL_CONFIG_NAME).write_text(yaml.dump(training_module_config))
    (tmp_path / DATA_CONFIG_NAME).write_text(
        yaml.dump({"history_mins": HISTORY_MINS, "forecast_mins": FORECAST_MINS})
    )
    (tmp_path / FULL_CONFIG_NAME).write_text(
        yaml.dump({"seed": 12345, "model": training_module_config})
    )

    torch.save({"state_dict": training_module.state_dict()}, tmp_path / "epoch=1-step=10.ckpt")

    return str(tmp_path)
