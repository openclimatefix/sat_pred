"""Tests for loading a model back out of its checkpoint directory"""

import os

import pytest
import torch

from sat_pred.constants import FULL_CONFIG_NAME
from sat_pred.load_model import get_model_from_checkpoints
from sat_pred.models.simvp_model import SimVP
from sat_pred.training_module import TrainingModule


def test_get_model_from_checkpoints(checkpoint_dir, tiny_model_config):
    """A saved checkpoint round-trips back to a model and its configs"""
    model, model_config, data_config, experiment_config_path = get_model_from_checkpoints(
        checkpoint_dir
    )

    # The lightning wrapper is discarded, leaving the torch model
    assert isinstance(model, torch.nn.Module)
    assert not isinstance(model, TrainingModule)
    assert isinstance(model, SimVP)

    # The config returned is the model config, not the training module which wraps it
    assert model_config == tiny_model_config

    assert data_config["history_mins"] > 0
    assert experiment_config_path == f"{checkpoint_dir}/{FULL_CONFIG_NAME}"


def test_get_model_from_checkpoints_without_experiment_config(checkpoint_dir):
    """Checkpoints saved before the full config was saved still load"""
    os.remove(f"{checkpoint_dir}/{FULL_CONFIG_NAME}")

    *_, experiment_config_path = get_model_from_checkpoints(checkpoint_dir)

    assert experiment_config_path is None


def test_get_model_from_checkpoints_restores_the_weights(checkpoint_dir):
    """The weights loaded are the ones which were saved"""
    checkpoint = torch.load(f"{checkpoint_dir}/epoch=1-step=10.ckpt", map_location="cpu")

    model, *_ = get_model_from_checkpoints(checkpoint_dir)

    for name, param in model.named_parameters():
        torch.testing.assert_close(param, checkpoint["state_dict"][f"model.{name}"])


def test_get_model_from_checkpoints_val_best(checkpoint_dir):
    """`val_best` picks between the best and the last checkpoint, and errors if it is missing"""
    # There is no last.ckpt in the fixture
    with pytest.raises(FileNotFoundError):
        get_model_from_checkpoints(checkpoint_dir, val_best=False)

    os.rename(f"{checkpoint_dir}/epoch=1-step=10.ckpt", f"{checkpoint_dir}/last.ckpt")

    model, *_ = get_model_from_checkpoints(checkpoint_dir, val_best=False)
    assert isinstance(model, torch.nn.Module)

    # ...and now there is no epoch*.ckpt to load as the best model
    with pytest.raises(ValueError, match="Found 0 checkpoints"):
        get_model_from_checkpoints(checkpoint_dir, val_best=True)
