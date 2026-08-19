"""Load a trained model, either from its checkpoint directory or from huggingface

The two sources hold the model differently. A checkpoint directory is what training writes: the
weights are inside a lightning checkpoint and `model_config.yaml` describes the `TrainingModule`
wrapping the model. Huggingface holds what `scripts/push_checkpoint_to_huggingface.py` pushed: the
weights as safetensors and a `model_config.yaml` describing the bare model. Same filename, different
content - see `get_model_from_huggingface`.

Both loaders return the model in whatever mode hydra instantiated it. Callers move it to their
device and call `.eval()` themselves.
"""

import os
from glob import glob

import hydra
import torch
import yaml
from huggingface_hub import snapshot_download
from safetensors.torch import load_model as load_model_from_safetensor

from sat_pred.constants import (
    DATA_CONFIG_NAME,
    FULL_CONFIG_NAME,
    MODEL_CONFIG_NAME,
    PYTORCH_WEIGHTS_NAME,
)


def _read_yaml_config(path: str) -> dict:
    """Read one of the YAML configs saved alongside a model"""
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


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
) -> tuple[torch.nn.Module, dict, dict, str]:
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
    training_module_config = _read_yaml_config(f"{checkpoint_dir_path}/{MODEL_CONFIG_NAME}")

    lightning_wrapped_model = hydra.utils.instantiate(training_module_config)

    checkpoint = torch.load(
        get_checkpoint_path(checkpoint_dir_path, val_best=val_best),
        map_location="cpu",
    )

    lightning_wrapped_model.load_state_dict(state_dict=checkpoint["state_dict"])

    # Discard the lightning wrapper on the model
    model = lightning_wrapped_model.model
    model_config = training_module_config["model"]

    data_config = _read_yaml_config(f"{checkpoint_dir_path}/{DATA_CONFIG_NAME}")

    experiment_config_path = f"{checkpoint_dir_path}/{FULL_CONFIG_NAME}"

    return model, model_config, data_config, experiment_config_path


def get_model_from_huggingface(
    repo_id: str,
    revision: str,
) -> tuple[torch.nn.Module, dict]:
    """Load a model from a huggingface model repo

    The repo holds what `scripts/push_checkpoint_to_huggingface.py` pushed. Note that its
    `model_config.yaml` describes the **bare model**, whereas the file of the same name in a
    checkpoint directory describes the lightning `TrainingModule` which wraps it - so the two
    loaders instantiate from configs of different shapes.

    The model is returned in whatever mode hydra instantiated it, matching
    `get_model_from_checkpoints`. Callers move it to their device and call `.eval()`.

    Args:
        repo_id: The huggingface model repo, e.g. "openclimatefix-models/cloudcasting_uk"
        revision: The commit hash, tag or branch to download. Pass a commit hash to pin the model -
            a branch moves under you

    Returns:
        tuple:
            model: The trained torch model
            data_config: The config of the data the model was trained on. This gives the history,
                forecast horizon, time resolution and channels the model expects, so the inputs
                built for it always match what it was trained to read
    """

    download_dir = snapshot_download(repo_id=repo_id, revision=revision)

    model_config = _read_yaml_config(f"{download_dir}/{MODEL_CONFIG_NAME}")

    model = hydra.utils.instantiate(model_config)

    load_model_from_safetensor(
        model,
        filename=f"{download_dir}/{PYTORCH_WEIGHTS_NAME}",
        strict=True,
    )

    data_config = _read_yaml_config(f"{download_dir}/{DATA_CONFIG_NAME}")

    return model, data_config
