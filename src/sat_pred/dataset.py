"""Dataset and DataModule for past and future satellite data"""


import atexit
import os
import shutil
import tempfile
from typing import TypedDict

import numpy as np
import pandas as pd
import xarray as xr
from lightning import LightningDataModule
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset
from ocf_data_sampler.load.utils import (
    check_time_unique_increasing,
    get_xr_data_array_from_xr_dataset,
    make_spatial_coords_increasing,
)
from ocf_data_sampler.select.find_contiguous_time_periods import find_contiguous_t0_periods
from ocf_data_sampler.torch_datasets.pvnet_dataset import PickleCacheMixin
from sat_pred.channels import ChannelConfigInput, parse_channel_config
from sat_pred.xr_tensorstore import open_zarr_paths


DataIndex = str | pd.Timestamp | int

# A [start_time, end_time] period which samples are taken from. Either bound can be None to leave
# that end of the period open
TimePeriod = list[str | None] | tuple[str | None, str | None]


class DataloaderArgs(TypedDict):
    batch_size: int
    sampler: None
    batch_sampler: None
    num_workers: int
    pin_memory: bool
    timeout: int
    worker_init_fn: None
    prefetch_factor: int | None
    persistent_workers: bool
    multiprocessing_context: str | None


def open_sat_data(zarr_path: str | list[str]) -> xr.DataArray:
    """Lazily opens the zarr store and validates data types.

    Args:
        zarr_path: Path(s) to the zarr file(s)
    """
    ds = open_zarr_paths(zarr_path, concat_dim="time")

    # The stores label the time dimension `time`, everything downstream of here uses `time_utc`
    ds = ds.rename({"time": "time_utc"})

    check_time_unique_increasing(ds.time_utc)
    ds = make_spatial_coords_increasing(ds, x_coord="x_geostationary", y_coord="y_geostationary")
    ds = ds.transpose("channel", "time_utc", "y_geostationary", "x_geostationary")
    da = get_xr_data_array_from_xr_dataset(ds)

    # The area definition which georeferences the images is stored on the dataset rather than on
    # the variable, so pulling the variable out drops it. It is kept here so that anything written
    # from this data can carry the geolocation with it
    da = da.assign_attrs(ds.attrs)

    if not np.issubdtype(da.dtype, np.floating):
        raise TypeError(f"Satellite data should be floating, not {da.dtype}")

    return da


def find_valid_t0_times(
    datetimes: pd.DatetimeIndex,
    history_mins: int,
    forecast_mins: int,
    sample_freq_mins: int,
) -> pd.DatetimeIndex:
    """Constuct an array of all t0 times which are valid considering the gaps in the sat data"""

    # Find periods of valid init-times
    contiguous_t0_periods = find_contiguous_t0_periods(
        datetimes=datetimes,
        interval_start=-pd.Timedelta(minutes=history_mins),
        interval_end=pd.Timedelta(minutes=forecast_mins),
        time_resolution=pd.Timedelta(minutes=sample_freq_mins),
    )

    valid_t0_times = []
    for _, row in contiguous_t0_periods.iterrows():
        valid_t0_times.append(
            pd.date_range(row["start_dt"], row["end_dt"], freq=f"{sample_freq_mins}min")
        )

    return pd.to_datetime(np.unique(np.concatenate(valid_t0_times)))


def mask_t0_time_periods(
    times: pd.DatetimeIndex,
    time_periods: list[tuple[str | None, str | None]],
) -> np.ndarray:
    """"Mask the given times to only those which fall within the given time periods.

    A `None` bound means the period is unbounded in that direction.

    Args:
        times: Array of times to filter
        time_periods: List of tuples specifying the start and end times for each period
    """
    if len(time_periods)==0:
        raise ValueError("At least one time period must be provided")

    mask = np.full(len(times), False)

    for start_time, end_time in time_periods:

        this_period_mask = np.full(len(times), True)

        # Inclusive of start_time, exclusive of end_time
        if start_time is not None:
            this_period_mask &= times >= np.datetime64(start_time)
        if end_time is not None:
            this_period_mask &= times < np.datetime64(end_time)

        mask |= this_period_mask

    return times[mask]


class SatelliteDataset(PickleCacheMixin, Dataset[tuple[NDArray[np.float32], NDArray[np.float32]]]):
    def __init__(
        self,
        zarr_path: list[str] | str,
        time_periods: list[TimePeriod],
        history_mins: int,
        forecast_mins: int,
        sample_freq_mins: int,
        channels: ChannelConfigInput,
        preshuffle: bool = False,
        seed: int | None = None,
    ):
        """A torch Dataset for loading past and future satellite data

        Args:
            zarr_path (list[str] | str): Path to the satellite data. Can be a string or list
            time_periods (list): List of [start_time, end_time] periods which samples are taken
                from. Either bound can be None to leave that end of the period open. Samples never
                span more than one period
            history_mins (int): How many minutes of history will be used as input features
            forecast_mins (int): How many minutes of future will be used as target features
            sample_freq_mins (int): The sample frequency to use for the satellite data
            channels: The channels to load, in the order they should appear in the samples, and
                the constants each one is clipped and z-scored with. Either a mapping of channel
                name to constants, or the path of a YAML file holding one - see
                `configs/datamodule/channels/`
            preshuffle (bool): Whether to shuffle the data - useful for validation.
                Defaults to False.
            seed (int | None): Seed used to shuffle the data if `preshuffle` is True. Set this to
                make the shuffled order reproducible between runs
        """
        # `PickleCacheMixin` sets up the pickle-cache state, so it has to run before anything
        # tries to pickle this dataset out to a dataloader worker
        super().__init__()

        self.channel_config = parse_channel_config(channels)

        # Load the sat zarr file or list of files
        da = open_sat_data(zarr_path)

        # Selecting by name puts the channels in the order the config lists them, whatever order
        # the store holds them in, so a model always reads the same channel at the same index
        da = da.sel(channel=self.channel_config.names)

        # Convert the satellite data to the given time frequency by selection
        mask = np.mod(da.time_utc.dt.minute, sample_freq_mins) == 0
        da = da.sel(time_utc=mask)

        # Find the valid t0 times within each period. This avoids trying to take samples where
        # there would be a missing timestamp in the sat data required for the sample. Since the
        # valid t0 times are found within each period separately, no sample spans two periods
        t0_times = find_valid_t0_times(
            pd.to_datetime(da.time_utc.values),
            history_mins,
            forecast_mins,
            sample_freq_mins,
        )
        t0_times = mask_t0_time_periods(t0_times, time_periods)

        if preshuffle:
            rng = np.random.default_rng(seed)
            t0_times = pd.to_datetime(rng.permutation(t0_times))

        self.history_delta = pd.Timedelta(minutes=history_mins)
        self.forecast_delta = pd.Timedelta(minutes=forecast_mins)
        self.sample_freq = pd.Timedelta(minutes=sample_freq_mins)
        self.normaliser = self.channel_config.normaliser
        self.da = da
        self.t0_times = t0_times

    def __setstate__(self, state: dict) -> None:
        """Rebuild a dataset sent to a dataloader worker

        `PickleCacheMixin` sends only the path of the presaved pickle when there is one, and reads
        the real state back from that file here. If the file has gone the mixin quietly leaves the
        dataset holding nothing but the path, which surfaces much later as a confusing
        `AttributeError` from inside a worker, so the missing file is caught here instead.
        """
        super().__setstate__(state)

        if self._pickle_path is not None and not hasattr(self, "t0_times"):
            raise FileNotFoundError(
                f"The presaved dataset pickle '{self._pickle_path}' no longer exists. It is "
                "deleted when training tears down, so this usually means another run sharing the "
                "same `dataset_pickle_dir` has finished and removed it."
            )

    def __len__(self) -> int:
        return len(self.t0_times)

    def _get_datetime(self, t0: np.datetime64) -> tuple[NDArray[np.float32], NDArray[np.float32]]:

        da_sel = self.da.sel(time_utc=slice(t0 - self.history_delta, t0 + self.forecast_delta))

        # Load the data eagerly so that the same chunks aren't loaded multiple times after we split
        # further
        da_sel = da_sel.compute()

        da_input = da_sel.sel(time_utc=slice(None, t0))
        da_target = da_sel.sel(time_utc=slice(t0 + self.sample_freq, None))

        # Convert to arrays of per-channel z-scores. The target keeps its NaNs so the missing
        # pixels can be masked out of the loss, but the model input cannot carry them
        X = self.normaliser.fill_missing(self.normaliser.normalise(da_input.values))
        y = self.normaliser.normalise(da_target.values)

        return X, y

    def __getitem__(self, key: DataIndex) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        if isinstance(key, int):
            t0 = self.t0_times[key]

        else:
            assert isinstance(key, str | np.datetime64 | pd.Timestamp)
            t0 = np.datetime64(key)
            if t0 not in self.t0_times:
                raise KeyError(
                    f"{t0} is not a valid init-time for this dataset. Either it falls outside the "
                    "time periods the dataset covers, or the satellite data is missing one of the "
                    "timestamps the sample would span."
                )

        return self._get_datetime(t0)


class SatelliteDataModule(LightningDataModule):
    def __init__(
        self,
        zarr_path: list[str] | str,
        history_mins: int,
        forecast_mins: int,
        sample_freq_mins: int,
        channels: ChannelConfigInput,
        batch_size: int = 16,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        train_periods: list[TimePeriod] | None = None,
        val_periods: list[TimePeriod] | None = None,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        seed: int | None = None,
        dataset_pickle_dir: str | None = None,
    ):
        """A lightning DataModule for loading past and future satellite data

        Args:
            zarr_path (list[str] | str): Path to the satellite data. Can be a string or list
            history_mins (int): How many minutes of history will be used as input features
            forecast_mins (int): How many minutes of future will be used as target features
            sample_freq_mins (int): The sample frequency to use for the satellite data
            channels: The channels to load, in the order they should appear in the samples, and
                the constants each one is clipped and z-scored with. Either a mapping of channel
                name to constants, or the path of a YAML file holding one - see
                `configs/datamodule/channels/`
            batch_size (int): Batch size. Defaults to 16.
            num_workers (int): Number of workers to use in multiprocess batch loading.
                Defaults to 0.
            prefetch_factor (int): Number of data to be prefetched at the end of each worker process
            train_periods (list): List of [start, end] date ranges the train samples are taken from
            val_periods (list): List of [start, end] date ranges the validation samples are taken
                from
            pin_memory (bool):  If True, the data loader will copy Tensors into device/CUDA
                pinned memory before returning them. Defaults to False.
            persistent_workers (bool): If True, the data loader will not shut down the worker
                processes after a dataset has been consumed once. This allows you to keep the
                workers Dataset instances alive. Defaults to False.
            seed (int | None): Seed used to shuffle the validation samples. Set this to make the
                validation set ordering reproducible between runs
            dataset_pickle_dir (str | None): Directory to presave the datasets into, so that
                starting a dataloader worker is a read from this directory rather than a fresh
                pickle of the dataset down the worker's pipe. A per-run subdirectory is made
                inside it and removed on teardown. `None` disables the cache
        """
        super().__init__()

        if train_periods is None:
            train_periods = [[None, None]]
        if val_periods is None:
            val_periods = [[None, None]]

        for period in [*train_periods, *val_periods]:
            assert len(period) == 2, f"Each period must be [start, end], got {period}"

        self.train_periods = train_periods
        self.val_periods = val_periods
        self.seed = seed

        # Made lazily on the first dataloader so that a run which never validates does not pay for
        # the val dataset, and so the val samples are only shuffled once per run
        self._datasets: dict[str, SatelliteDataset] = {}

        self.dataset_pickle_dir = dataset_pickle_dir
        self._pickle_run_dir: str | None = None

        self.zarr_path = zarr_path
        self.history_mins = history_mins
        self.forecast_mins = forecast_mins
        self.sample_freq_mins = sample_freq_mins

        self._common_dataloader_kwargs = DataloaderArgs(
            batch_size=batch_size,
            sampler=None,
            batch_sampler=None,
            num_workers=num_workers,
            pin_memory=pin_memory,
            timeout=0,
            worker_init_fn=None,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            multiprocessing_context="forkserver" if num_workers>0 else None,
        )

        # Parsed once here so a bad config fails before any data is loaded
        self.channel_config = parse_channel_config(channels)

    def _make_dataset(
        self, time_periods: list[TimePeriod], preshuffle: bool = False
    ) -> SatelliteDataset:
        return SatelliteDataset(
            zarr_path=self.zarr_path,
            time_periods=time_periods,
            history_mins=self.history_mins,
            forecast_mins=self.forecast_mins,
            sample_freq_mins=self.sample_freq_mins,
            preshuffle=preshuffle,
            seed=self.seed,
            channels=self.channel_config,
        )

    def _get_dataset(self, split: str) -> SatelliteDataset:
        """Return the dataset for a split, making and presaving it the first time it is asked for

        The dataset is cached because lightning asks for the dataloaders again whenever it reloads
        them, and rebuilding means reopening every zarr store and refinding the valid t0 times.
        """
        if split in self._datasets:
            return self._datasets[split]

        if split == "train":
            dataset = self._make_dataset(self.train_periods)
        else:
            dataset = self._make_dataset(self.val_periods, preshuffle=True)

        if self.dataset_pickle_dir is not None:
            dataset.presave_pickle(f"{self._get_pickle_run_dir()}/{split}_dataset.pkl")

        self._datasets[split] = dataset
        return dataset

    def _get_pickle_run_dir(self) -> str:
        """Return this run's private directory for presaved datasets, making it if needed

        Each run gets its own subdirectory. Sharing one path between runs would mean a run
        overwriting the pickle another run's workers are still restoring from, and deleting it out
        from under them when it tears down.
        """
        if self._pickle_run_dir is None:
            os.makedirs(self.dataset_pickle_dir, exist_ok=True)
            # Absolute, because a worker's working directory is not guaranteed to be the one the
            # cache directory was configured relative to
            self._pickle_run_dir = os.path.abspath(
                tempfile.mkdtemp(prefix="sat_pred-", dir=self.dataset_pickle_dir)
            )
            # Lightning does not always reach `teardown` when a run dies part way through, e.g. on
            # a CUDA OOM, which would leave the presaved datasets behind for good
            atexit.register(self.teardown)
        return self._pickle_run_dir

    def teardown(self, stage: str | None = None) -> None:
        """Delete this run's presaved datasets"""
        if self._pickle_run_dir is not None:
            shutil.rmtree(self._pickle_run_dir, ignore_errors=True)
            self._pickle_run_dir = None
            # The datasets still point at the deleted pickles, so they can no longer be sent to a
            # worker. Dropping them means a later dataloader rebuilds and presaves from scratch
            self._datasets.clear()

    def train_dataloader(self) -> DataLoader[tuple[NDArray[np.float32], NDArray[np.float32]]]:
        """Construct train dataloader"""
        dataset = self._get_dataset("train")
        # Drop the final part-batch so all training batches are the same size
        return DataLoader(dataset, shuffle=True, drop_last=True, **self._common_dataloader_kwargs)

    def val_dataloader(self) -> DataLoader[tuple[NDArray[np.float32], NDArray[np.float32]]]:
        """Construct validation dataloader"""
        dataset = self._get_dataset("val")
        return DataLoader(dataset, shuffle=False, drop_last=False, **self._common_dataloader_kwargs)