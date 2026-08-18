"""Load a model from its checkpoint directory"""

import os
from glob import glob
from pathlib import Path

import hydra
import torch
import yaml

from sat_pred.constants import DATA_CONFIG_NAME, FULL_CONFIG_NAME, MODEL_CONFIG_NAME


def get_checkpoint_path(checkpoint_dir_path: str, val_best: bool = True) -> str:
    """Find the checkpoint file to load from a checkpoint directory

    Args:
        checkpoint_dir_path: Path to the checkpoint directory
        val_best: Whether to use the best performing checkpoint found during training, else uses
            the last checkpoint saved during training

    Returns:
        Path to the checkpoint file
    """

    if not val_best:
        return f"{checkpoint_dir_path}/last.ckpt"

    # Only one epoch (best) saved per model
    files = glob(f"{checkpoint_dir_path}/epoch*.ckpt")
    if len(files) != 1:
        raise ValueError(
            f"Found {len(files)} checkpoints @ {checkpoint_dir_path}/epoch*.ckpt. Expected one."
        )
    return files[0]


def get_model_from_checkpoints(
    checkpoint_dir_path: str,
    val_best: bool = True,
):
    """Load a model from its checkpoint directory

    Args:
        checkpoint_dir_path: Path to the checkpoint directory
        val_best: Whether to use the best performing checkpoint found during training, else uses
            the last checkpoint saved during training

    Returns:
        tuple:
            model: The trained torch model, with the lightning wrapper discarded
            model_config: The config of the torch model
            data_config: The config of the data the model was trained on
            experiment_config_path: Path to the full hydra config of the training run. This is
                None for models trained before these configs were saved
    """

    # Load the config of the lightning module used in training
    training_module_config = yaml.safe_load(
        Path(f"{checkpoint_dir_path}/{MODEL_CONFIG_NAME}").read_text()
    )

    lightning_wrapped_model = hydra.utils.instantiate(training_module_config)

    checkpoint = torch.load(
        get_checkpoint_path(checkpoint_dir_path, val_best=val_best), map_location="cpu"
    )

    lightning_wrapped_model.load_state_dict(state_dict=checkpoint["state_dict"])

    # Discard the lightning wrapper on the model
    model = lightning_wrapped_model.model
    model_config = training_module_config["model"]

    # Check for data config
    data_config = yaml.safe_load(Path(f"{checkpoint_dir_path}/{DATA_CONFIG_NAME}").read_text())

    # Check for the full config of the training run. This is only saved for newer models
    experiment_config_path = f"{checkpoint_dir_path}/{FULL_CONFIG_NAME}"
    if not os.path.isfile(experiment_config_path):
        experiment_config_path = None

    return model, model_config, data_config, experiment_config_path
