"""Tests for loading a trained model back out of a checkpoint directory or huggingface"""

import os

import pytest
import torch
from safetensors import safe_open

from sat_pred.constants import FULL_CONFIG_NAME, PYTORCH_WEIGHTS_NAME
from sat_pred.load_model import get_model_from_checkpoints, get_model_from_huggingface
from sat_pred.models.simvp_model import SimVP
from sat_pred.training_module import TrainingModule

from .conftest import HISTORY_MINS, IMAGE_SIZE

# Stand-ins for the repo and pinned commit a caller would name
HF_REPO_ID = "some-org/some-model"
HF_REVISION = "8b1a9953c4611296a827abf8c47804d7"


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


@pytest.fixture
def downloaded_huggingface_dir(huggingface_dir, monkeypatch) -> str:
    """`huggingface_dir`, with the download it stands in for patched out"""

    def fake_snapshot_download(repo_id: str, revision: str) -> str:
        assert (repo_id, revision) == (HF_REPO_ID, HF_REVISION)
        return huggingface_dir

    monkeypatch.setattr("sat_pred.load_model.snapshot_download", fake_snapshot_download)

    return huggingface_dir


def test_get_model_from_huggingface(downloaded_huggingface_dir):
    """A model pushed to huggingface round-trips back to a model, data config and grid"""
    model, data_config, spatial_grid = get_model_from_huggingface(HF_REPO_ID, HF_REVISION)

    # The huggingface repo holds the bare model, whereas a checkpoint directory holds the training
    # module wrapping it - so a loader which conflated the two would fail here
    assert isinstance(model, SimVP)
    assert not isinstance(model, TrainingModule)

    assert data_config["history_mins"] == HISTORY_MINS

    assert len(spatial_grid.x_geostationary) == IMAGE_SIZE
    assert len(spatial_grid.y_geostationary) == IMAGE_SIZE

    # `strict=True` already catches a key mismatch. This catches the weights never being loaded at
    # all, which leaves the model randomly initialised and forecasting silently wrong
    weights_path = f"{downloaded_huggingface_dir}/{PYTORCH_WEIGHTS_NAME}"
    with safe_open(weights_path, framework="pt") as weights_file:
        for name, param in model.named_parameters():
            torch.testing.assert_close(param, weights_file.get_tensor(name))
