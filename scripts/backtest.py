"""A script to run backtest for sat_pred

See example_backtest_data_config.yaml for the expected format of the input data config yaml file.


Example command to run the backtest
```
python backtest.py \
    --checkpoint /home/james/repos/sat_pred/checkpoints/xlq6w7qj \
    --input-data-paths example_backtest_data_config.yaml \
    --output-zarr-path /path/to/output.zarr \
    --start-datetime "2022-01-01 00:00" \
    --end-datetime "2022-12-31 23:30" \
    --device-name 'cuda:0' \
    --num-workers 8 \
    --batch-size 4
```

You can also get help on the command line arguments with:
```
python scripts/backtest.py --help
```

"""

from collections import deque
from numpy.typing import NDArray
import torch
import dask.array
import numpy as np
import pandas as pd
import tensorstore as ts
import xarray as xr
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import tempfile
import typer
import yaml
import zarr

from sat_pred.channels import ChannelNormaliser, parse_channel_config
from sat_pred.dataset import SatelliteDataset, TimePeriod
from sat_pred.load_model import get_model_from_checkpoints



chunks_dict = {
    "init_time_utc": 1,
    "step": -1,
    "x_geostationary": 25,
    "y_geostationary": 25,
    "channel": -1,
}
shards_dict = {
    "init_time_utc": 1,
    "step": -1,
    "x_geostationary": -1,
    "y_geostationary": -1,
    "channel": -1,
}

compressor = zarr.codecs.BloscCodec(
    cname="lz4", 
    clevel=5,
    shuffle="shuffle",
    blocksize=0,
)

app = typer.Typer(pretty_exceptions_show_locals=False)


class DeviceNormaliser:
    """The channel normalisation constants, held as tensors on the model's device

    The backtest normalises on the same device the model runs on rather than in the dataloader
    workers. The samples then reach this process as the raw values the store holds, which for a
    float16 store is half the bytes of the normalised float32 the workers used to send, and none of
    the arithmetic lands on the main process, which is the one the model is waiting on.

    The arithmetic is the same as `ChannelNormaliser` does in numpy, in the same float32, so the
    predictions are unchanged.
    """

    def __init__(self, normaliser: ChannelNormaliser, device: torch.device) -> None:
        """Move a `ChannelNormaliser`'s constants onto a device

        Args:
            normaliser: The normaliser holding the constants of each channel
            device: The device the samples are normalised on
        """

        def constant(values: NDArray[np.float32]) -> torch.Tensor:
            # `ChannelNormaliser` shapes its constants to broadcast against a single sample. The
            # samples here are batched, so they need one more leading dimension
            return torch.as_tensor(values, dtype=torch.float32, device=device).unsqueeze(0)

        self.mean = constant(normaliser.mean)
        self.std = constant(normaliser.std)
        self.clip_min = constant(normaliser.clip_min)
        self.clip_max = constant(normaliser.clip_max)
        self.missing_fill_value = constant(normaliser.missing_fill_value)

    def normalise(self, values: torch.Tensor) -> torch.Tensor:
        """Clip each channel to its physical range, z-score it, and fill the missing pixels"""
        values = values.to(torch.float32).clamp(self.clip_min, self.clip_max)
        values = (values - self.mean) / self.std
        # Clipping leaves the missing pixels as NaN, and convolutions would spread each one across
        # everything downstream of it
        return torch.where(values.isnan(), self.missing_fill_value, values)

    def denormalise(self, values: torch.Tensor) -> torch.Tensor:
        """Convert z-scores back to physical units"""
        return values * self.std + self.mean


class MLModel:
    """A trained model, and the sample shape it was trained on, loaded from a checkpoint"""

    def __init__(self, checkpoint_dir_path: str, device: torch.device) -> None:
        """Load a trained model from its checkpoint directory

        Args:
            checkpoint_dir_path: Path of the checkpoint directory
            device: The torch device to run the model on
        """

        model, _, data_config, _ = get_model_from_checkpoints(checkpoint_dir_path, val_best=True)

        # `.eval()` because a freshly instantiated module is in train mode
        self.model = model.to(device).eval()
        self.device = device
        self.checkpoint_dir_path = checkpoint_dir_path

        # The shape of the samples the model was trained on. These are unpacked from the data
        # config rather than the model config so the backtest samples are built exactly the way the
        # training samples were, and so nothing downstream has to know the data config's layout
        self.sample_freq_mins = data_config['sample_freq_mins']
        self.history_mins = data_config['history_mins']
        self.forecast_mins = data_config['forecast_mins']
        self.channels = data_config['channels']

        self.normaliser = DeviceNormaliser(
            parse_channel_config(self.channels).normaliser, device
        )

    @torch.no_grad()
    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """Predict from a batch of raw satellite images

        Args:
            X: The raw images the store holds, shaped (batch, channel, time, y, x), on the CPU

        Returns:
            The predictions in the physical units of each channel, as float16 on the CPU. Float16
            because that is what the output store holds, so casting here rather than at write time
            halves both the copy off the device and the bytes handed to the writer
        """
        X = X.to(self.device, non_blocking=True)
        y_hat = self.model(self.normaliser.normalise(X))
        return self.normaliser.denormalise(y_hat).to(torch.float16).cpu()


def get_satellite_paths(input_data_paths: str) -> list[str]:
    """Load the paths of the satellite zarrs from a data config YAML file

    Args:
        input_data_paths: Path of the YAML file. See example_backtest_data_config.yaml for the
            expected format
    """

    with open(input_data_paths) as f:
        config = yaml.safe_load(f)

    if "satellite" not in config:
        raise ValueError(f"No `satellite` key found in the data config: {input_data_paths}")

    paths = config["satellite"]

    return [paths] if isinstance(paths, str) else list(paths)


def create_zarr_encoding(
    ds: xr.Dataset, 
    chunks_dict: dict[str, int], 
    shards_dict: dict[str, int], 
    compressor: zarr.codecs.BloscCodec,
) -> dict:
    """Build the zarr encoding for a dataset holding a single data variable

    Args:
        ds: The dataset the encoding is built for
        chunks_dict: Chunk size for each dimension. -1 means the full length of that dimension
        shards_dict: Shard size for each dimension. -1 means the full length of that dimension,
            rounded up to a whole number of chunks
        compressor: The compressor to store the data variable with
    """

    assert len(ds.data_vars)==1
    assert set(chunks_dict)==set(shards_dict)
    data_var = next(iter(ds.data_vars))
    assert set(chunks_dict)==set(ds[data_var].dims)

    chunks_dict = chunks_dict.copy()
    shards_dict = shards_dict.copy()

    # Set the chunk and shard sizes for each dimension, replacing -1 with the full length of that 
    # dimension
    for k in chunks_dict:
        if chunks_dict[k] == -1: 
            chunks_dict[k] = len(ds[k])
        if shards_dict[k] == -1: 
            shards_dict[k] = len(ds[k])
            if shards_dict[k] % chunks_dict[k] != 0:
                shards_dict[k] = (shards_dict[k] // chunks_dict[k] + 1)*chunks_dict[k]

    chunk_shape = [chunks_dict[k] for k in ds[data_var].dims]
    shard_shape = [shards_dict[k] for k in ds[data_var].dims]

    # Set the encoding for the dimensions
    encoding = {d: {"chunks": (len(ds[d]),)} for d in ds.dims if d != "init_time_utc"}
    # The init-times are on the half hour, so seconds are more resolution than they need. The unit
    # must not be finer than the resolution pandas gives the times - xarray aligns the divisor down
    # to that resolution, and a unit finer than it divides by zero and silently writes NaT
    encoding["init_time_utc"] = {
        "dtype": "int",
        "units": "seconds since 1970-01-01",
        "calendar": "proleptic_gregorian",
        "chunks": (1000,),
    }

    # Set the encoding for the data variable
    encoding[data_var] = {
        "compressors": (compressor,),
        "chunks": chunk_shape,
        "shards": shard_shape,
        "dtype": "float16",
    }

    return encoding


def backtest_collate_fn(
    samples: list[tuple[torch.Tensor, np.datetime64]],
) -> tuple[torch.Tensor, NDArray[np.datetime64]]:
    """Compile the model inputs and their init-times into a batch

    The inputs are stacked into a torch tensor rather than a numpy array because a worker hands a
    tensor to the main process through shared memory, while a numpy array is pickled down the
    worker's socket and copied out of it by the main process - the one thread the model is waiting
    on. At these sample sizes that copy costs more than everything else the main process does.
    """

    X_all = torch.stack([X for X, _ in samples])
    ts = np.array([t for _, t in samples], dtype='datetime64[ns]')

    return X_all, ts


class BacktestSatelliteDataset(SatelliteDataset):
    def __init__(
        self,
        zarr_path: list[str] | str,
        time_periods: list[TimePeriod],
        history_mins: int,
        sample_freq_mins: int,
        channels,
    ):
        """A torch Dataset for loading model inputs and the init-time they are a forecast from

        Args:
            zarr_path: Path to the satellite data. Can be a string or list
            time_periods: List of [start_time, end_time] periods which init-times are taken from.
                Either bound can be None to leave that end of the period open
            history_mins: How many minutes of history will be used as input features
            sample_freq_mins: The sample frequency to use for the satellite data
            channels: The channels the model was trained on, in order, and the constants each one
                is normalised with. Either a mapping of channel name to constants, or the path of a
                YAML file holding one
        """

        # No target is loaded - the backtest only makes predictions, it does not score them
        super().__init__(
            zarr_path=zarr_path,
            time_periods=time_periods,
            history_mins=history_mins,
            forecast_mins=0,
            sample_freq_mins=sample_freq_mins,
            channels=channels,
        )

        # Only forecast on the half hour
        self.t0_times = self.t0_times[self.t0_times.minute%30==0]

    def _get_datetime(self, t0: np.datetime64) -> tuple[torch.Tensor, np.datetime64]:
        da_input = self.da.sel(time_utc=slice(t0 - self.history_delta, t0))

        # Load the data eagerly so that the same chunks aren't loaded multiple times after we split
        # further
        da_input = da_input.compute()

        # The raw images are handed on as they are stored, and normalised on the model's device
        # instead - see `DeviceNormaliser`. Normalising here would widen a float16 store's samples
        # to float32 before they are sent to the main process, doubling the bytes it has to copy
        return torch.from_numpy(da_input.values), t0


def create_prediction_store(
    output_zarr_path: str,
    init_times: pd.DatetimeIndex,
    coords: dict[str, NDArray],
    attrs: dict,
) -> ts.TensorStore:
    """Lay the whole output zarr out up front and open its data variable for writing

    Every init-time the backtest will predict for is known before it starts, so the store is made at
    its full size and each batch of predictions is written into its own region of it. That is what
    lets the predictions be written by tensorstore, which compresses and writes in its own threads
    and so costs the loop feeding the model almost nothing. Appending instead means every write
    growing the store, which only zarr-python can do, and which costs several times more per
    init-time than everything else the backtest does put together.

    An init-time which is never written keeps the store's NaN fill value, so a backtest which dies
    part way through leaves a store whose missing predictions are missing rather than plausible.

    Args:
        output_zarr_path: Path of the zarr store to create
        init_times: Every init-time which will be predicted for, in the order they are written
        coords: The coordinates shared by every prediction - the channels, steps, and the two
            spatial coordinates, in the order they are stored
        attrs: Attributes to store on the output dataset

    Returns:
        The store's data variable, opened for writing
    """

    shape = (len(init_times), *(len(values) for values in coords.values()))

    # Dask-backed, so `compute=False` writes the metadata and the coordinates and leaves the
    # predictions themselves unwritten
    template = xr.DataArray(
        dask.array.zeros(shape, dtype=np.float16, chunks=(1, *shape[1:])),
        dims=["init_time_utc", *coords],
        coords={"init_time_utc": init_times, **coords},
    ).to_dataset(name="sat_pred")
    template.attrs.update(attrs)

    encoding = create_zarr_encoding(
        template, chunks_dict=chunks_dict, shards_dict=shards_dict, compressor=compressor,
    )
    # Consolidated metadata is not part of the zarr-3 spec, and writing it warns
    template.to_zarr(
        output_zarr_path, mode="w", encoding=encoding, consolidated=False, compute=False,
    )

    return ts.open(
        {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": os.path.join(output_zarr_path, "sat_pred")},
        },
        write=True,
        read=False,
    ).result()


def run_backtest(
    model: MLModel,
    dataset: BacktestSatelliteDataset,
    output_zarr_path: str,
    batch_size: int,
    num_workers: int,
    writes_in_flight: int = 4,
) -> None:
    """Predict for every init-time in the dataset and save the predictions to a zarr store

    The predictions are saved in the physical units of each channel, not the z-scores the model
    works in.

    Args:
        model: The model to make the predictions with
        dataset: The dataset of model inputs to predict from
        output_zarr_path: Path of the zarr store to save the predictions to. The store is laid out
            at its full size before the first batch - see `create_prediction_store`
        batch_size: Number of init-times predicted in each forward pass of the model
        num_workers: Number of workers used to load the satellite data
        writes_in_flight: How many batches of predictions may be being written at once. Each one
            waiting to be written holds a batch of predictions in memory, and waiting for the oldest
            of them is what stops the model running ahead of the writes indefinitely
    """

    backtest_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=backtest_collate_fn,
        drop_last=False,
        # Pinned batches copy to the device without going through a staging buffer first, and the
        # copy can overlap the model. The pinning itself happens in the dataloader's own thread
        pin_memory=num_workers>0,
        # The satellite data is read through tensorstore, which uses threads internally and aborts
        # the process if it is forked. The workers have to be started from a clean process instead
        multiprocessing_context="forkserver" if num_workers>0 else None,
    )

    attrs_dict = dict(dataset.da.attrs)
    attrs_dict["model_checkpoint"] = model.checkpoint_dir_path
    steps = pd.timedelta_range(
        start=f"{model.sample_freq_mins}min",
        end=f"{model.forecast_mins}min",
        freq=f"{model.sample_freq_mins}min",
    )
    prediction_store = create_prediction_store(
        output_zarr_path,
        init_times=dataset.t0_times,
        coords={
            "channel": dataset.da.channel.values,
            "step": steps,
            "y_geostationary": dataset.da.y_geostationary.values,
            "x_geostationary": dataset.da.x_geostationary.values,
        },
        attrs=attrs_dict,
    )

    # Each write keeps a reference to the predictions it is writing, because tensorstore reads them
    # from the array it was handed rather than copying them first
    writes: deque[tuple[ts.WriteFutures, NDArray[np.float16]]] = deque()
    written = 0

    for X, t in tqdm(backtest_dataloader, total=len(backtest_dataloader)):

        # The model predicts z-scores. They are converted back to the physical units of each
        # channel - brightness temperatures in Kelvin, reflectances in percent - so the store
        # can be read without knowing which normalisation constants the model was trained with
        y_hat = model(X).numpy()

        region = slice(written, written + len(y_hat))
        # The dataloader is not shuffled, so each batch holds the next block of init-times. Checked
        # rather than assumed, because writing a batch to the wrong region of the store would
        # mislabel predictions rather than fail
        assert (dataset.t0_times[region] == t).all()

        # This returns as soon as the write is queued. Tensorstore compresses and writes in its own
        # threads, so the writing overlaps the loading and predicting rather than interrupting it
        writes.append((prediction_store[region].write(y_hat), y_hat))
        written = region.stop

        while len(writes) > writes_in_flight:
            writes.popleft()[0].result()

    for write, _ in writes:
        write.result()


@app.command()
def backtest(
    input_data_paths: str = typer.Option(..., "--input-data-paths"),
    output_zarr_path: str = typer.Option(..., "--output-zarr-path"),
    checkpoint: str = typer.Option(..., "--checkpoint"),
    start_datetime: str = typer.Option(..., "--start-datetime"),
    end_datetime: str = typer.Option(..., "--end-datetime"),
    device_name: str = typer.Option(..., "--device-name"),
    num_workers: int = typer.Option(..., "--num-workers"),
    batch_size: int = typer.Option(..., "--batch-size"),
    dataset_pickle_dir: str = typer.Option(None, "--dataset-pickle-dir"),
) -> None:
    """Run a backtest of the model checkpoint and save the predictions to zarr

    Args:
        input_data_paths: Path of the YAML file holding the satellite zarr paths. See
            example_backtest_data_config.yaml for the expected format
        output_zarr_path: Path of the zarr store to save the predictions to
        checkpoint: Path of the checkpoint directory to run the backtest for
        start_datetime: The first init-time predicted for. The history the prediction is made from
            is read from the satellite data before this time
        end_datetime: Init-times from this time onwards are not predicted for
        device_name: The torch device to run the model on, e.g. "cuda:0" or "cpu"
        num_workers: Number of workers used to load the satellite data
        batch_size: Number of init-times predicted in each forward pass of the model
        dataset_pickle_dir: Directory to presave the dataset into, so that starting a dataloader
            worker is a read from this directory rather than a fresh pickle of the dataset down the
            worker's pipe. A run-specific subdirectory is made inside it and removed when the run
            ends. Defaults to the system temp directory, and is unused when `num_workers` is 0
    """

    if os.path.exists(output_zarr_path):
        raise FileExistsError(f"There is already something saved at {output_zarr_path}")

    model = MLModel(checkpoint, torch.device(device_name))

    dataset = BacktestSatelliteDataset(
        zarr_path=get_satellite_paths(input_data_paths),
        time_periods=[[start_datetime, end_datetime]],
        history_mins=model.history_mins,
        sample_freq_mins=model.sample_freq_mins,
        channels=model.channels,
    )

    # Without this the run would quietly do nothing and write no store at all, which is easy to
    # hit by asking for a period the satellite data does not cover
    if len(dataset) == 0:
        raise ValueError(
            f"There are no init-times to predict for between {start_datetime} and {end_datetime}"
        )

    if dataset_pickle_dir is not None:
        os.makedirs(dataset_pickle_dir, exist_ok=True)

    # The run gets its own subdirectory, which is removed however the run ends. Absolute, because a
    # worker's working directory is not guaranteed to be the one this was configured relative to
    with tempfile.TemporaryDirectory(prefix="sat_pred-", dir=dataset_pickle_dir) as pickle_run_dir:

        # Presaving means each worker reads the dataset from this file rather than being sent a
        # fresh pickle of it down its pipe, the same as training does. With no workers there is
        # nothing to send it to, and the file would only be written and deleted again
        if num_workers > 0:
            dataset.presave_pickle(f"{os.path.abspath(pickle_run_dir)}/backtest_dataset.pkl")

        run_backtest(
            model=model,
            dataset=dataset,
            output_zarr_path=output_zarr_path,
            batch_size=batch_size,
            num_workers=num_workers,
        )


if __name__=="__main__":
    app()
