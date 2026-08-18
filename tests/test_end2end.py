"""End-to-end training test"""

import hydra
import lightning
import torch

from sat_pred.dataset import SatelliteDataModule
from tests.conftest import (
    CHANNELS_PATH,
    DATA_FREQ_MINS,
    FORECAST_MINS,
    HISTORY_MINS,
)

TRAIN_PERIODS = [["2020-01-01 00:00", "2020-01-03 00:00"]]
VAL_PERIODS = [["2020-01-05 00:00", "2020-01-06 00:00"]]


def test_model_trainer_fit(sat_zarr_path, training_module_config):
    """Train the model end to end for a couple of batches"""

    datamodule = SatelliteDataModule(
        zarr_path=sat_zarr_path,
        history_mins=HISTORY_MINS,
        forecast_mins=FORECAST_MINS,
        sample_freq_mins=DATA_FREQ_MINS,
        batch_size=2,
        num_workers=0,
        train_periods=TRAIN_PERIODS,
        val_periods=VAL_PERIODS,
        channels=CHANNELS_PATH,
        seed=0,
    )

    training_module = hydra.utils.instantiate(training_module_config)

    trainer = lightning.Trainer(
        max_epochs=2,
        limit_train_batches=2,
        limit_val_batches=2,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    trainer.fit(model=training_module, datamodule=datamodule)

    assert trainer.state.finished
    assert trainer.global_step == 4

    # The losses were logged under the names the callbacks and optimizer monitor
    assert "MAE/val" in trainer.callback_metrics
    assert "MSE/val" in trainer.callback_metrics
    assert "MAE/train" in trainer.callback_metrics

    assert torch.isfinite(trainer.callback_metrics["MAE/val"])


def test_model_trainer_validate(sat_zarr_path, training_module_config):
    """Run the validation loop on its own"""

    datamodule = SatelliteDataModule(
        zarr_path=sat_zarr_path,
        history_mins=HISTORY_MINS,
        forecast_mins=FORECAST_MINS,
        sample_freq_mins=DATA_FREQ_MINS,
        batch_size=2,
        num_workers=0,
        train_periods=TRAIN_PERIODS,
        val_periods=VAL_PERIODS,
        channels=CHANNELS_PATH,
    )

    training_module = hydra.utils.instantiate(training_module_config)

    trainer = lightning.Trainer(
        limit_val_batches=2,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    (results,) = trainer.validate(model=training_module, datamodule=datamodule)

    assert torch.isfinite(torch.tensor(results["MAE/val"]))
