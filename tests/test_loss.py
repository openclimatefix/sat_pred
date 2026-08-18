"""Tests for the custom loss functions"""

import torch

from sat_pred.loss import MultiscaleMAE

SCALES = [(1, 1, 1), (2, 4, 4)]
SHAPE = (2, 3, 4, 8, 8)


def test_multiscale_mae_matches_mae_at_full_resolution():
    """With only the unit scale the loss is a plain MAE"""
    loss_function = MultiscaleMAE(scales=[(1, 1, 1)])

    y_hat = torch.zeros(SHAPE)
    y = torch.full(SHAPE, 2.0)

    assert loss_function(y_hat, y).item() == 2.0


def test_multiscale_mae_ignores_missing_pixels():
    """Missing pixels arrive as NaN and must not poison the loss"""
    loss_function = MultiscaleMAE(scales=SCALES)

    y_hat = torch.zeros(SHAPE)
    y = torch.full(SHAPE, 2.0)
    y[:, :, 0] = torch.nan

    loss = loss_function(y_hat, y)

    assert torch.isfinite(loss)
    assert loss.item() == 2.0


def test_multiscale_mae_does_not_produce_nan_gradients():
    """Masked-out pixels must not leak NaN back through the gradients"""
    loss_function = MultiscaleMAE(scales=SCALES)

    y_hat = torch.zeros(SHAPE, requires_grad=True)
    y = torch.full(SHAPE, 2.0)
    y[:, :, 0] = torch.nan

    loss_function(y_hat, y).backward()

    assert not torch.isnan(y_hat.grad).any()


def test_multiscale_mae_does_not_modify_its_target():
    """The target is shared with the other losses, so it must be left alone"""
    loss_function = MultiscaleMAE(scales=SCALES)

    y = torch.full(SHAPE, 2.0)
    y[:, :, 0] = torch.nan

    loss_function(torch.zeros(SHAPE), y)

    assert (y[:, :, 1:] == 2.0).all()
    assert torch.isnan(y[:, :, 0]).all()
