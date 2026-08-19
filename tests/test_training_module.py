"""Tests for the lightning module which wraps the model"""

import hydra
import numpy as np
import torch

import sat_pred.training_module as training_module
from sat_pred.channels import ChannelNormalisation
from sat_pred.training_module import (
    VIDEO_CLIP_SIGMA,
    VIDEO_RANGE_BUFFER,
    TrainingModule,
    greyscale_range_label,
    upload_video,
    video_greyscale_limits,
)
from tests.conftest import (
    IMAGE_SIZE,
    NUM_CHANNELS,
    NUM_FORECAST_STEPS,
    NUM_HISTORY_STEPS,
)

VIDEO_STEPS = 2
VIDEO_HEIGHT = 4
VIDEO_WIDTH = 6


def run_upload_video(
    monkeypatch,
    values: torch.Tensor,
    vmin: float = -VIDEO_CLIP_SIGMA,
    vmax: float = VIDEO_CLIP_SIGMA,
) -> np.ndarray:
    """Run `upload_video` and return the frames it would have sent to wandb"""
    monkeypatch.setattr(training_module.wandb, "Video", lambda frames, **_kwargs: frames)

    captured = {}

    class FakeRun:
        def log(self, data, **_kwargs):
            captured.update(data)

    upload_video(values, values, "video_name", FakeRun(), 0, vmin, vmax, channel_num=0)

    return captured["video_name"]


def gradient_along(axis: str) -> torch.Tensor:
    """A (channel, time, y, x) tensor whose values ramp from 0 to 1 along one spatial axis"""
    y_ramp = torch.linspace(0, 1, VIDEO_HEIGHT)[:, None].expand(VIDEO_HEIGHT, VIDEO_WIDTH)
    x_ramp = torch.linspace(0, 1, VIDEO_WIDTH)[None, :].expand(VIDEO_HEIGHT, VIDEO_WIDTH)
    ramp = y_ramp if axis == "y" else x_ramp
    return ramp.expand(1, VIDEO_STEPS, VIDEO_HEIGHT, VIDEO_WIDTH).clone()


def test_video_puts_north_at_the_top(monkeypatch):
    """The y coordinate increases northwards, so it must be flipped to draw north at the top"""
    # Values ramp from 0 at the lowest y (south) to 1 at the highest y (north)
    frames = run_upload_video(monkeypatch, gradient_along("y"))

    assert frames.shape == (VIDEO_STEPS, 3, VIDEO_HEIGHT, VIDEO_WIDTH * 2)

    top_row, bottom_row = frames[0, 0, 0, :], frames[0, 0, -1, :]
    assert top_row.min() > bottom_row.max(), "north should be drawn at the top of the video"


def test_video_is_not_mirrored_east_west(monkeypatch):
    """The x coordinate increases eastwards, which already matches drawing left to right"""
    # Values ramp from 0 at the lowest x (west) to 1 at the highest x (east)
    frames = run_upload_video(monkeypatch, gradient_along("x"))

    # The two halves of the video are the prediction and the truth. Check within one of them
    left_half = frames[0, 0, :, :VIDEO_WIDTH]
    assert left_half[:, 0].max() < left_half[:, -1].min(), "east should be drawn on the right"


def test_video_draws_the_given_window_across_the_greyscale_range(monkeypatch):
    """The window fills the greyscale range, and values beyond it are clipped"""
    values = torch.tensor([-2.0, -1.0, 0.0, 1.0])
    values = values.reshape(1, 1, 1, 4).expand(1, VIDEO_STEPS, 1, 4).clone()

    frames = run_upload_video(monkeypatch, values, vmin=-1.0, vmax=1.0)

    # The two halves of the video are the prediction and the truth. Check within one of them
    row = frames[0, 0, 0, :4]

    assert list(row) == [0, 0, 127, 255]


def test_video_draws_missing_pixels_black(monkeypatch):
    """Missing pixels have no value to draw, and must not become arbitrary noise"""
    values = torch.zeros(1, VIDEO_STEPS, VIDEO_HEIGHT, VIDEO_WIDTH)
    values[:, :, 0, 0] = torch.nan

    frames = run_upload_video(monkeypatch, values, vmin=-1.0, vmax=1.0)

    assert not np.isnan(frames).any()

    # The video is flipped to put north at the top, so the first row of the data is drawn last
    assert frames[0, 0, -1, 0] == 0
    # Zero sits mid window, so the missing pixel is distinguishable from the rest
    assert frames[0, 0, 1, 1] == 127


def test_greyscale_limits_span_the_whole_target():
    """A window taken from the target itself spends the greyscale on the values it contains"""
    values = torch.zeros(VIDEO_STEPS, VIDEO_HEIGHT, VIDEO_WIDTH)
    # The extremes are put in different frames, so limits taken from one frame would miss one
    values[0, 0, 0] = -1.0
    values[-1, 0, 0] = 3.0

    vmin, vmax = video_greyscale_limits(values)

    buffer = VIDEO_RANGE_BUFFER * 4.0
    assert (vmin, vmax) == (-1.0 - buffer, 3.0 + buffer)


def test_greyscale_limits_ignore_missing_pixels():
    """Missing pixels are NaN in the target, and must not drag the limits to NaN"""
    values = torch.zeros(VIDEO_STEPS, VIDEO_HEIGHT, VIDEO_WIDTH)
    values[0, 0, 0] = torch.nan
    values[0, 1, 1] = 2.0

    vmin, vmax = video_greyscale_limits(values)

    buffer = VIDEO_RANGE_BUFFER * 2.0
    assert (vmin, vmax) == (-buffer, 2.0 + buffer)


def test_greyscale_limits_fall_back_when_the_target_has_no_range():
    """An all-missing or flat target has no range to stretch across the greyscale"""
    all_missing = torch.full((VIDEO_STEPS, VIDEO_HEIGHT, VIDEO_WIDTH), torch.nan)
    flat = torch.zeros(VIDEO_STEPS, VIDEO_HEIGHT, VIDEO_WIDTH)

    assert video_greyscale_limits(all_missing) == (-VIDEO_CLIP_SIGMA, VIDEO_CLIP_SIGMA)
    assert video_greyscale_limits(flat) == (-VIDEO_CLIP_SIGMA, VIDEO_CLIP_SIGMA)


def test_greyscale_range_label_reports_physical_values():
    """wandb shows no scale on a video, so the window has to be in the name"""
    label = greyscale_range_label(
        -3.0,
        3.0,
        ChannelNormalisation(
            mean=263.5, std=16.0, clip_min=190, clip_max=320, missing_value=174
        ),
    )

    assert label == "[216-312]"


def test_losses_ignore_missing_pixels(training_module_config):
    """Missing pixels, which arrive as NaN in the target, are excluded from the losses"""
    training_module: TrainingModule = hydra.utils.instantiate(training_module_config)

    y_hat = torch.zeros(2, 3, 4, 5, 5)
    y = torch.zeros(2, 3, 4, 5, 5)

    # Without any missing data the errors are zero
    losses = training_module._calculate_common_losses(y, y_hat)
    assert losses["MAE"].item() == 0
    assert losses["MSE"].item() == 0

    # Marking part of the target as missing leaves the remaining errors zero, rather than
    # poisoning the whole loss with NaN
    y[:, :, 0] = torch.nan
    losses = training_module._calculate_common_losses(y, y_hat)
    assert losses["MAE"].item() == 0
    assert losses["MSE"].item() == 0

    # Real errors in the unmasked pixels are still counted
    y[:, :, 1] = 1
    losses = training_module._calculate_common_losses(y, y_hat)
    assert losses["MAE"].item() > 0


def test_losses_do_not_produce_nan_gradients(training_module_config):
    """Masked-out pixels must not leak NaN back through the gradients"""
    training_module: TrainingModule = hydra.utils.instantiate(training_module_config)

    y_hat = torch.zeros(2, 3, 4, 5, 5, requires_grad=True)
    y = torch.ones(2, 3, 4, 5, 5)
    y[:, :, 0] = torch.nan

    training_module._calculate_common_losses(y, y_hat)["MAE"].backward()

    assert not torch.isnan(y_hat.grad).any()


def test_losses_are_nan_when_the_whole_target_is_missing(training_module_config):
    """An all-missing target gives a NaN loss, which training and validation then handle"""
    training_module: TrainingModule = hydra.utils.instantiate(training_module_config)

    y_hat = torch.zeros(2, 3, 4, 5, 5)
    y = torch.full((2, 3, 4, 5, 5), torch.nan)

    losses = training_module._calculate_common_losses(y, y_hat)

    assert np.isnan(losses["MAE"].item())
    assert np.isnan(losses["MSE"].item())


def run_training_step(monkeypatch, training_module_config, y) -> tuple:
    """Run a training step against the given target and return what it logged and returned"""
    training_module: TrainingModule = hydra.utils.instantiate(training_module_config)

    logged = {}
    monkeypatch.setattr(
        TrainingModule, "log_dict", lambda _self, metrics, **_kwargs: logged.update(metrics)
    )

    X = torch.zeros(1, NUM_CHANNELS, NUM_HISTORY_STEPS, IMAGE_SIZE, IMAGE_SIZE)

    return logged, training_module.training_step((X, y), batch_idx=0)


def test_training_step_logs_metrics(monkeypatch, training_module_config):
    """A normal batch logs every metric, and as tensors so the GPU is not synced"""
    y = torch.zeros(1, NUM_CHANNELS, NUM_FORECAST_STEPS, IMAGE_SIZE, IMAGE_SIZE)

    logged, train_loss = run_training_step(monkeypatch, training_module_config, y)

    assert set(logged) == {"MAE/train", "MSE/train"}
    assert all(torch.is_tensor(v) for v in logged.values())
    assert train_loss is not None


def test_training_step_logs_nothing_when_the_whole_target_is_missing(
    monkeypatch, training_module_config
):
    """A NaN loss is skipped rather than logged, where it would drag the epoch mean to NaN"""
    y = torch.full((1, NUM_CHANNELS, NUM_FORECAST_STEPS, IMAGE_SIZE, IMAGE_SIZE), torch.nan)

    logged, train_loss = run_training_step(monkeypatch, training_module_config, y)

    assert logged == {}
    assert train_loss is None
