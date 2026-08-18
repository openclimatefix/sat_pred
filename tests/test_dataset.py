"""Tests for the satellite dataset and datamodule"""

import os

import numpy as np
import pandas as pd
import pytest

from sat_pred.dataset import SatelliteDataModule, SatelliteDataset, find_valid_t0_times
from tests.conftest import (
    CHANNELS_PATH,
    DATA_FREQ_MINS,
    FORECAST_MINS,
    GAP_END,
    GAP_START,
    HISTORY_MINS,
    IMAGE_SIZE,
    NUM_CHANNELS,
    NUM_FORECAST_STEPS,
    NUM_HISTORY_STEPS,
)

# Two periods either side of the gap in the synthetic data
PERIOD_1 = ["2020-01-01 00:00", "2020-01-03 00:00"]
PERIOD_2 = ["2020-01-05 00:00", "2020-01-06 00:00"]

SAMPLE_KWARGS = {
    "history_mins": HISTORY_MINS,
    "forecast_mins": FORECAST_MINS,
    "sample_freq_mins": DATA_FREQ_MINS,
    "channels": CHANNELS_PATH,
}


def make_dataset(zarr_path, time_periods, **kwargs) -> SatelliteDataset:
    return SatelliteDataset(
        zarr_path=zarr_path, time_periods=time_periods, **SAMPLE_KWARGS, **kwargs
    )


def test_find_valid_t0_times_excludes_gaps(sat_datetimes):
    """t0 times which would need missing timestamps are not returned"""
    t0_times = find_valid_t0_times(
        sat_datetimes, HISTORY_MINS, FORECAST_MINS, DATA_FREQ_MINS
    )

    assert len(t0_times) > 0

    for t0 in t0_times:
        window = pd.date_range(
            t0 - pd.Timedelta(minutes=HISTORY_MINS),
            t0 + pd.Timedelta(minutes=FORECAST_MINS),
            freq=f"{DATA_FREQ_MINS}min",
        )
        assert window.isin(sat_datetimes).all(), f"sample at {t0} needs missing timestamps"

    # No t0 can sit inside the gap in the data
    assert not ((t0_times >= GAP_START) & (t0_times <= GAP_END)).any()


def test_dataset_length_and_sample_shape(sat_zarr_path):
    """Samples have the expected shape and dtype"""
    dataset = make_dataset(sat_zarr_path, [PERIOD_1])

    assert len(dataset) == len(dataset.t0_times) > 0

    X, y = dataset[0]

    assert X.shape == (NUM_CHANNELS, NUM_HISTORY_STEPS, IMAGE_SIZE, IMAGE_SIZE)
    assert y.shape == (NUM_CHANNELS, NUM_FORECAST_STEPS, IMAGE_SIZE, IMAGE_SIZE)
    assert X.dtype == np.float32
    assert y.dtype == np.float32


def test_dataset_accepts_a_list_of_zarr_paths(sat_zarr_path):
    """The zarr path can be given as a list, which is how the configs supply it"""
    dataset_from_str = make_dataset(sat_zarr_path, [PERIOD_1])
    dataset_from_list = make_dataset([sat_zarr_path], [PERIOD_1])

    assert dataset_from_list.t0_times.equals(dataset_from_str.t0_times)


def test_dataset_indexing_by_datetime(sat_zarr_path):
    """A sample can be selected by t0 datetime as well as by position"""
    dataset = make_dataset(sat_zarr_path, [PERIOD_1])

    t0 = dataset.t0_times[5]

    X_int, y_int = dataset[5]
    X_datetime, y_datetime = dataset[t0]

    np.testing.assert_array_equal(X_int, X_datetime)
    np.testing.assert_array_equal(y_int, y_datetime)

    # A t0 which isn't valid for this dataset is rejected
    with pytest.raises(AssertionError):
        dataset[GAP_START]


def test_the_channel_config_selects_and_orders_the_channels(sat_zarr_path, channel_config):
    """Samples hold exactly the channels the config names, in the order it names them"""
    # Deliberately a subset, and in a different order to the store
    channels = {name: channel_config[name].model_dump() for name in ["WV_073", "IR_016"]}

    dataset = SatelliteDataset(
        zarr_path=sat_zarr_path,
        time_periods=[PERIOD_1],
        history_mins=HISTORY_MINS,
        forecast_mins=FORECAST_MINS,
        sample_freq_mins=DATA_FREQ_MINS,
        channels=channels,
    )

    assert list(dataset.da.channel.values) == ["WV_073", "IR_016"]

    X, _ = dataset[0]
    assert X.shape == (2, NUM_HISTORY_STEPS, IMAGE_SIZE, IMAGE_SIZE)

    # Each channel is normalised with its own constants, not the store's first two
    np.testing.assert_allclose(
        dataset.normaliser.mean.ravel(),
        [channel_config["WV_073"].mean, channel_config["IR_016"].mean],
    )


def test_a_channel_the_data_does_not_have_is_an_error(sat_zarr_path, channel_config):
    """Asking for a channel which is not in the store fails rather than being dropped"""
    channels = {"NOT_A_CHANNEL": channel_config["IR_016"].model_dump()}

    with pytest.raises(KeyError):
        SatelliteDataset(
            zarr_path=sat_zarr_path,
            time_periods=[PERIOD_1],
            history_mins=HISTORY_MINS,
            forecast_mins=FORECAST_MINS,
            sample_freq_mins=DATA_FREQ_MINS,
            channels=channels,
        )


def test_samples_are_normalised(sat_zarr_path, channel_config):
    """Samples come out as per-channel z-scores of the clipped physical values"""
    dataset = make_dataset(sat_zarr_path, [PERIOD_1])

    X, y = dataset[0]

    # The synthetic data spans each channel's full clipped range, so every channel should come out
    # roughly centred on its own mean rather than on the raw physical values
    for channel_num, channel_name in enumerate(dataset.da.channel.values):
        values = channel_config[channel_name]
        expected_min = (values.clip_min - values.mean) / values.std
        expected_max = (values.clip_max - values.mean) / values.std

        for sample in (X, y):
            assert sample[channel_num].min() >= expected_min
            assert sample[channel_num].max() <= expected_max


def test_normalisation_clips_out_of_range_values(sat_zarr_path, channel_config):
    """Values outside a channel's physical range are pulled back to its bounds"""
    dataset = make_dataset(sat_zarr_path, [PERIOD_1])

    # One physically impossible pixel per channel, at either end of the range
    raw = np.stack(
        [
            np.full((2, 2, 2), values.clip_min - 1000)
            for values in [channel_config[c] for c in dataset.da.channel.values]
        ]
    )
    normalised = dataset.normaliser.normalise(raw)

    expected = (dataset.normaliser.clip_min - dataset.normaliser.mean) / dataset.normaliser.std
    np.testing.assert_allclose(normalised, np.broadcast_to(expected, raw.shape), rtol=1e-6)


def test_missing_pixels_are_filled_in_the_input_but_not_the_target(sat_zarr_path_with_nans):
    """The model input cannot carry NaNs, but the target keeps them so they can be masked"""
    dataset = make_dataset(sat_zarr_path_with_nans, [PERIOD_1])

    # The 11th timestamp of the data is all-NaN, so it lands in the target of some early sample
    samples = [dataset[i] for i in range(12)]
    assert any(np.isnan(y).any() for _, y in samples), "expected a target with missing pixels"

    for X, _ in samples:
        assert not np.isnan(X).any(), "the model input must not contain NaNs"

    # Wherever the input was filled, it holds that channel's configured missing value
    for channel_num in range(NUM_CHANNELS):
        fill_value = dataset.normaliser.missing_fill_value[channel_num, 0, 0, 0]
        assert any((X[channel_num] == fill_value).any() for X, _ in samples)


def test_multiple_periods_are_the_union_of_the_single_periods(sat_zarr_path):
    """Passing two periods gives the samples of both"""
    dataset_1 = make_dataset(sat_zarr_path, [PERIOD_1])
    dataset_2 = make_dataset(sat_zarr_path, [PERIOD_2])
    dataset_both = make_dataset(sat_zarr_path, [PERIOD_1, PERIOD_2])

    assert len(dataset_1) > 0 and len(dataset_2) > 0
    assert len(dataset_both) == len(dataset_1) + len(dataset_2)
    assert dataset_both.t0_times.equals(dataset_1.t0_times.union(dataset_2.t0_times))


def test_overlapping_periods_are_deduplicated(sat_zarr_path):
    """Overlapping periods give each sample once, in time order"""
    dataset = make_dataset(
        sat_zarr_path,
        [["2020-01-01 00:00", "2020-01-02 12:00"], ["2020-01-02 00:00", "2020-01-03 00:00"]],
    )
    dataset_merged = make_dataset(sat_zarr_path, [PERIOD_1])

    assert dataset.t0_times.is_unique
    assert dataset.t0_times.is_monotonic_increasing
    assert dataset.t0_times.equals(dataset_merged.t0_times)


def test_open_ended_periods(sat_zarr_path):
    """A period bound of None means no bound on that end"""
    dataset_all = make_dataset(sat_zarr_path, [[None, None]])
    dataset_bounded = make_dataset(sat_zarr_path, [PERIOD_1])

    assert len(dataset_all) > len(dataset_bounded)


def test_preshuffle_is_reproducible_for_a_given_seed(sat_zarr_path):
    """The same seed shuffles the same way, and shuffling only changes the order"""
    unshuffled = make_dataset(sat_zarr_path, [PERIOD_1])
    shuffled = make_dataset(sat_zarr_path, [PERIOD_1], preshuffle=True, seed=42)
    shuffled_again = make_dataset(sat_zarr_path, [PERIOD_1], preshuffle=True, seed=42)
    shuffled_other_seed = make_dataset(sat_zarr_path, [PERIOD_1], preshuffle=True, seed=43)

    assert shuffled.t0_times.equals(shuffled_again.t0_times)
    assert not shuffled.t0_times.equals(shuffled_other_seed.t0_times)

    assert not shuffled.t0_times.is_monotonic_increasing
    assert shuffled.t0_times.sort_values().equals(unshuffled.t0_times)


def test_datamodule_dataloaders(sat_zarr_path):
    """The datamodule passes its periods and seed through, and drops the last train batch only"""
    datamodule = SatelliteDataModule(
        zarr_path=sat_zarr_path,
        **SAMPLE_KWARGS,
        batch_size=2,
        train_periods=[PERIOD_1],
        val_periods=[PERIOD_2],
        seed=7,
    )

    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()

    assert train_dataloader.drop_last
    assert not val_dataloader.drop_last

    assert len(train_dataloader.dataset) == len(make_dataset(sat_zarr_path, [PERIOD_1]))
    assert len(val_dataloader.dataset) == len(make_dataset(sat_zarr_path, [PERIOD_2]))

    # The validation set is shuffled, using the seed the datamodule was given
    expected_val_t0s = make_dataset(sat_zarr_path, [PERIOD_2], preshuffle=True, seed=7).t0_times
    assert val_dataloader.dataset.t0_times.equals(expected_val_t0s)

    # The train set is left in time order - the dataloader shuffles it instead
    assert train_dataloader.dataset.t0_times.is_monotonic_increasing


def test_datamodule_defaults_to_all_the_data(sat_zarr_path):
    """With no periods given the datamodule uses everything available"""
    datamodule = SatelliteDataModule(
        zarr_path=sat_zarr_path, **SAMPLE_KWARGS, batch_size=2
    )

    assert len(datamodule.train_dataloader().dataset) == len(
        make_dataset(sat_zarr_path, [[None, None]])
    )


def test_datamodule_batches(sat_zarr_path):
    """Batches come out of the dataloader with the batch dimension added"""
    datamodule = SatelliteDataModule(
        zarr_path=sat_zarr_path,
        **SAMPLE_KWARGS,
        batch_size=2,
        train_periods=[PERIOD_1],
        val_periods=[PERIOD_2],
    )

    X, y = next(iter(datamodule.train_dataloader()))

    assert X.shape == (2, NUM_CHANNELS, NUM_HISTORY_STEPS, IMAGE_SIZE, IMAGE_SIZE)
    assert y.shape == (2, NUM_CHANNELS, NUM_FORECAST_STEPS, IMAGE_SIZE, IMAGE_SIZE)


def test_datamodule_reuses_its_datasets(sat_zarr_path):
    """Asking for a dataloader twice does not rebuild the dataset behind it"""
    datamodule = SatelliteDataModule(
        zarr_path=sat_zarr_path, **SAMPLE_KWARGS, batch_size=2, seed=7
    )

    assert datamodule.train_dataloader().dataset is datamodule.train_dataloader().dataset
    assert datamodule.val_dataloader().dataset is datamodule.val_dataloader().dataset


def test_presaved_datasets_are_sent_to_workers_by_reference(sat_zarr_path, tmp_path):
    """With a pickle dir set, a dataset pickles to just a path but restores in full

    This is what makes starting a dataloader worker cheap - the worker reads the presaved file
    rather than having the whole dataset pushed down its pipe.
    """
    import pickle

    datamodule = SatelliteDataModule(
        zarr_path=sat_zarr_path,
        **SAMPLE_KWARGS,
        batch_size=2,
        train_periods=[PERIOD_1],
        dataset_pickle_dir=str(tmp_path / "cache"),
    )

    dataset = datamodule.train_dataloader().dataset

    # The blob a worker is sent holds the path and nothing else
    blob = pickle.dumps(dataset)
    assert len(blob) < len(pickle.dumps(make_dataset(sat_zarr_path, [PERIOD_1])))

    restored = pickle.loads(blob)
    assert restored.t0_times.equals(dataset.t0_times)
    np.testing.assert_array_equal(restored[0][0], dataset[0][0])


def test_teardown_removes_the_presaved_datasets(sat_zarr_path, tmp_path):
    """The presaved datasets belong to one run and do not outlive it"""
    import pickle

    cache_dir = tmp_path / "cache"
    datamodule = SatelliteDataModule(
        zarr_path=sat_zarr_path,
        **SAMPLE_KWARGS,
        batch_size=2,
        train_periods=[PERIOD_1],
        dataset_pickle_dir=str(cache_dir),
    )

    dataset = datamodule.train_dataloader().dataset
    blob = pickle.dumps(dataset)
    assert list(cache_dir.rglob("*.pkl"))

    datamodule.teardown()

    assert not list(cache_dir.rglob("*.pkl"))

    # A worker started against the deleted file says so, rather than failing later on a dataset
    # which quietly has no data in it
    with pytest.raises(FileNotFoundError):
        pickle.loads(blob)


def test_each_datamodule_presaves_to_its_own_directory(sat_zarr_path, tmp_path):
    """Two runs sharing a pickle dir must not overwrite or delete each other's datasets"""
    cache_dir = tmp_path / "cache"

    datamodules = [
        SatelliteDataModule(
            zarr_path=sat_zarr_path,
            **SAMPLE_KWARGS,
            batch_size=2,
            train_periods=[PERIOD_1],
            dataset_pickle_dir=str(cache_dir),
        )
        for _ in range(2)
    ]
    datasets = [dm.train_dataloader().dataset for dm in datamodules]

    assert datasets[0]._pickle_path != datasets[1]._pickle_path

    # One run tearing down leaves the other run's dataset alone
    datamodules[0].teardown()
    assert not os.path.exists(datasets[0]._pickle_path)
    assert os.path.exists(datasets[1]._pickle_path)
