import torch
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
from numcodecs import Blosc

from sat_pred.channels import parse_channel_config
from sat_pred.dataset import find_valid_t0_times, open_sat_data
from sat_pred.load_model import get_model_from_checkpoints


checkpoint = "/home/jamesfulton/repos/sat_pred/checkpoints/ob9v9128"
save_dir = "/mnt/disks/sat_preds/simvp_preds"
compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLModel:

    def __init__(self, checkpoint_dir_path: str) -> None:

        
        model, model_config, data_config, _ = get_model_from_checkpoints(checkpoint)

        self.model = model.to(DEVICE)
        self.history_mins = (model_config['history_len'] - 1) * 15
        self.model_config = model_config
        self.data_config = data_config
        self.checkpoint_dir_path = checkpoint_dir_path


    def __call__(self, X):
        # The input X is a numpy array with shape (batch_size, channels, time, height, width)
        X = torch.Tensor(X).to(DEVICE)
        
        with torch.no_grad():
            y_hat = self.model(X).cpu().numpy()

        return y_hat
    


def backtest_collate_fn(
    samples: list,
):
    # Create empty stores for the compiled batch
    X_all = np.empty((len(samples), *samples[0][0].shape), dtype=np.float32)

    # Fill the stores with the samples
    ts = []
    for i, (X, t) in enumerate(samples):
        X_all[i] = X
        ts.append(t)
    return X_all, pd.to_datetime(ts)
    

DataIndex = str | datetime | pd.Timestamp | int

class BacktestSatelliteDataset(Dataset):
    def __init__(
        self,
        zarr_path: list[str] | str,
        start_time: str | None,
        end_time: str | None,
        history_mins: int,
        sample_freq_mins: int,
        channels,
    ):
        """A torch Dataset for loading past and future satellite data

        Args:
            zarr_path: Path to the satellite data. Can be a string or list
            start_time: The satellite data is filtered to exclude timestamps before this
            end_time: The satellite data is filtered to exclude timestamps after this
            history_mins: How many minutes of history will be used as input features
            sample_freq_mins: The sample frequency to use for the satellite data
            channels: The channels the model was trained on, in order, and the constants each one
                is normalised with. Either a mapping of channel name to constants, or the path of a
                YAML file holding one
        """

        # Load the sat zarr file or list of files and slice the data to the given period
        self.da = open_sat_data(zarr_path).sel(time_utc=slice(start_time, end_time))

        # Convert the satellite data to the given time frequency by selection
        mask = np.mod(self.da.time_utc.dt.minute, 15) == 0
        self.da = self.da.sel(time_utc=mask)

        # Find the valid t0 times for the available data. This avoids trying to take samples where
        # there would be a missing timestamp in the sat data required for the sample
        self.t0_times = self._find_t0_times(
            pd.DatetimeIndex(self.da.time_utc), history_mins, sample_freq_mins
        )

        # Only do 30 minute intervals
        self.t0_times = self.t0_times[self.t0_times.minute%30==0]

        self.history_mins = history_mins
        self.sample_freq_mins = sample_freq_mins
        self.channel_config = parse_channel_config(channels)
        self.normaliser = self.channel_config.normaliser

        # Selected by name so the channels arrive in the order the model was trained to read them
        self.da = self.da.sel(channel=self.channel_config.names)

    @staticmethod
    def _find_t0_times(
        date_range: pd.DatetimeIndex, 
        history_mins: int, 
        sample_freq_mins: int
    ) -> pd.DatetimeIndex:
        return find_valid_t0_times(date_range, history_mins, 0, sample_freq_mins)

    def __len__(self):
        return len(self.t0_times)

    def _get_datetime(self, t0: datetime):
        da_input = self.da.sel(time_utc=slice(t0 - timedelta(minutes=self.history_mins), t0))

        # Load the data eagerly so that the same chunks aren't loaded multiple times after we split
        # further
        da_input = da_input.compute()

        # Convert to an array of per-channel z-scores. The model cannot consume NaNs, so the
        # missing pixels are filled the same way they are during training
        X = self.normaliser.fill_missing(self.normaliser.normalise(da_input.values))

        return X, t0

    def __getitem__(self, key: DataIndex):
        if isinstance(key, int):
            t0 = self.t0_times[key]

        else:
            assert isinstance(key, str | datetime | pd.Timestamp)
            t0 = pd.Timestamp(key)
            assert t0 in self.t0_times

        return self._get_datetime(t0)



def run_backtest(
    model: MLModel,
    dataset: BacktestSatelliteDataset,
    batch_size: int = 1,
    num_workers: int = 0,
    batch_limit: int | None = None,
    agg_batches: int = 1
) -> None:
    """Calculate the scoreboard metrics for the given model on the validation dataset.

    Args:
        model (AbstractModel): The model to score.
        valid_dataset (ValidationSatelliteDataset): The validation dataset to score the model on.
        batch_size (int, optional): Defaults to 1.
        num_workers (int, optional): Defaults to 0.
        batch_limit (int | None, optional): Defaults to None. Stop after this many batches.
            For testing purposes only.

    """

    backtest_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=backtest_collate_fn,
        drop_last=False,
    )

    # we probably want to accumulate metrics here instead of taking the mean of means!
    loop_steps = len(backtest_dataloader) if batch_limit is None else batch_limit

    da_y_hats = []
    save_batch_num = 0

    attrs_dict = {k:v for k,v in dataset.da.attrs.items()}
    attrs_dict["model_checkpoint"] = model.checkpoint_dir_path

    for i, (X, t) in tqdm(enumerate(backtest_dataloader), total=loop_steps):
        
        y_hat = model(X)
        init_times = pd.DatetimeIndex(t)
        steps = pd.timedelta_range("15min", periods=y_hat.shape[2], freq="15min")

        da_y_hat = xr.DataArray(
            y_hat, 
            dims=["init_time", "channel", "step", "y_geostationary", "x_geostationary"],
            coords={
                "init_time": init_times,
                "channel": dataset.da.channel,
                "step": steps,
                "y_geostationary": dataset.da.y_geostationary,
                "x_geostationary": dataset.da.x_geostationary,
            }
        ).chunk(
            {
                "init_time": 1,
                "channel":-1,
                "step":-1, 
                "y_geostationary": 100, 
                "x_geostationary": 100,
            }
        )

        da_y_hats.append(da_y_hat)

        del da_y_hat
        
        if len(da_y_hats)==agg_batches or i==loop_steps-1:

            da_y_hats = xr.concat(da_y_hats, dim="init_time")

            da_y_hats.attrs = attrs_dict

            da_y_hats = da_y_hats.to_dataset(name="sat_pred")
            
            da_y_hats.to_zarr(
                f"{save_dir}/part_{save_batch_num}.zarr", 
                mode="w",
                encoding={var: {'compressor': compressor} for var in da_y_hats.data_vars},
            )
            
            save_batch_num += 1
            da_y_hats = []
            

        if batch_limit is not None and i == batch_limit:
            break


if __name__=="__main__":


    os.makedirs(save_dir, exist_ok=False)

    model = MLModel(checkpoint)

    dataset = BacktestSatelliteDataset(
        zarr_path=[
            "/mnt/disks/all_data/sat/2019_nonhrv.zarr",
            "/mnt/disks/all_data/sat/2020_nonhrv.zarr",
            "/mnt/disks/all_data/sat/2021_nonhrv.zarr",
            "/mnt/disks/all_data/sat/2022_nonhrv.zarr",
            "/mnt/disks/all_data/sat/2023_nonhrv.zarr",
        ],
        start_time=None, 
        end_time=None,
        history_mins=model.history_mins,
        sample_freq_mins=15,
        channels=model.data_config['channels'],
    )

    run_backtest(
        model=model,
        dataset=dataset,
        batch_size=4,
        num_workers=12,
        batch_limit=None,
        agg_batches=8,
    )
