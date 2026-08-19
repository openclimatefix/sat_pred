"""Tests for the channel config and the normalisers built from it"""

import numpy as np
import pytest
import torch
import yaml
from pydantic import ValidationError

from sat_pred.channels import ChannelConfig, TorchChannelNormaliser, parse_channel_config

CONSTANTS = {
    "IR_108": {
        "mean": 260.0, "std": 10.0, "clip_min": 200.0, "clip_max": 320.0,
        "missing_value": 190.0,
    },
    "VIS008": {
        "mean": 10.0, "std": 5.0, "clip_min": 0.0, "clip_max": 100.0,
        "missing_value": -5.0,
    },
}


def test_a_config_can_come_from_a_mapping_or_a_path(tmp_path):
    """Hydra passes a mapping, but a path to a YAML file works too"""
    path = tmp_path / "channels.yaml"
    path.write_text(yaml.dump(CONSTANTS))

    from_mapping = parse_channel_config(CONSTANTS)
    from_path = parse_channel_config(str(path))

    assert from_mapping == from_path
    assert parse_channel_config(from_mapping) is from_mapping


def test_unusable_constants_are_rejected():
    """A std of zero or an inverted clip range would silently corrupt the data"""
    with pytest.raises(ValidationError):
        ChannelConfig.model_validate({"IR_108": {**CONSTANTS["IR_108"], "std": 0.0}})

    with pytest.raises(ValidationError, match="clip_min"):
        ChannelConfig.model_validate({"IR_108": {**CONSTANTS["IR_108"], "clip_min": 400.0}})


def test_an_empty_config_is_rejected():
    """A config which names no channels would give samples with no data in them"""
    with pytest.raises(ValidationError, match="at least one channel"):
        ChannelConfig.model_validate({})


def test_an_unknown_channel_is_named_in_the_error():
    """Asking for a channel the config does not hold says which one is missing"""
    config = parse_channel_config(CONSTANTS)

    with pytest.raises(KeyError, match="HRV"):
        config["HRV"]


def test_the_config_order_is_the_sample_order():
    """The order the channels are written in is the order the normaliser applies them in"""
    forwards = parse_channel_config({"IR_108": CONSTANTS["IR_108"], "VIS008": CONSTANTS["VIS008"]})
    backwards = parse_channel_config({"VIS008": CONSTANTS["VIS008"], "IR_108": CONSTANTS["IR_108"]})

    assert forwards.names == ["IR_108", "VIS008"]
    assert backwards.names == ["VIS008", "IR_108"]

    np.testing.assert_array_equal(forwards.normaliser.mean[::-1], backwards.normaliser.mean)


def test_normalise_clips_then_z_scores():
    """Each channel is clipped to its physical range and then centred on its own mean"""
    normaliser = parse_channel_config(CONSTANTS).normaliser

    # One value at the channel mean, one below clip_min, one above clip_max
    values = np.array([[[[260.0, 100.0, 400.0]]], [[[10.0, -50.0, 500.0]]]])

    normalised = normaliser.normalise(values)

    np.testing.assert_allclose(
        normalised,
        [
            [[[0.0, (200 - 260) / 10, (320 - 260) / 10]]],
            [[[0.0, (0 - 10) / 5, (100 - 10) / 5]]],
        ],
        rtol=1e-6,
    )
    assert normalised.dtype == np.float32


def test_normalise_keeps_missing_pixels():
    """Missing pixels stay NaN so the loss can mask them out later"""
    normaliser = parse_channel_config({"IR_108": CONSTANTS["IR_108"]}).normaliser

    normalised = normaliser.normalise(np.array([[[[260.0, np.nan]]]]))

    assert normalised[0, 0, 0, 0] == 0
    assert np.isnan(normalised[0, 0, 0, 1])


def test_denormalise_inverts_normalise():
    """Round tripping a value already inside the clip range gives it back"""
    normaliser = parse_channel_config(CONSTANTS).normaliser

    values = np.array([[[[260.0, 280.0]]], [[[10.0, 40.0]]]])

    np.testing.assert_allclose(
        normaliser.denormalise(normaliser.normalise(values)), values, rtol=1e-5
    )


def test_fill_missing_uses_the_configured_missing_value():
    """The model input has no NaNs, and each channel is filled with its own configured value"""
    normaliser = parse_channel_config(CONSTANTS).normaliser

    values = np.array([[[[260.0, np.nan]]], [[[10.0, np.nan]]]])

    filled = normaliser.fill_missing(normaliser.normalise(values))

    assert not np.isnan(filled).any()
    assert filled.dtype == np.float32

    # The z-scores of the configured 190.0 K and -5.0 %
    np.testing.assert_allclose(
        filled[:, 0, 0, 1], [(190 - 260) / 10, (-5 - 10) / 5], rtol=1e-6
    )


def test_the_missing_value_is_not_clipped():
    """A missing value below clip_min must stay below it, or it could not be told apart"""
    normaliser = parse_channel_config(CONSTANTS).normaliser

    clip_min_z = (normaliser.clip_min - normaliser.mean) / normaliser.std

    assert (normaliser.missing_fill_value < clip_min_z).all()


def test_the_missing_value_may_sit_inside_the_clip_range():
    """Filling gaps with the channel mean is a valid choice, so it is not rejected"""
    constants = {"IR_108": {**CONSTANTS["IR_108"], "missing_value": 260.0}}

    normaliser = parse_channel_config(constants).normaliser

    assert normaliser.missing_fill_value.item() == 0.0


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_the_torch_normaliser_matches_the_numpy_one(dtype):
    """The two implementations must agree, or the backtest and inference pathways would diverge

    Float64 input is the one case where they do not: numpy clips in float64 and casts to float32 at
    the end, whereas torch casts first and clips in float32. The satellite stores hold float32 or
    float16, which is what this covers.
    """
    numpy_normaliser = parse_channel_config(CONSTANTS).normaliser
    torch_normaliser = TorchChannelNormaliser(numpy_normaliser, torch.device("cpu"))

    # Below clip_min, inside the range, above clip_max, and missing - for each channel of each of
    # the two samples in the batch
    values = np.array(
        [
            [[[[100.0, 260.0, 400.0, np.nan]]], [[[-50.0, 10.0, 500.0, np.nan]]]],
            [[[[199.0, 300.0, 321.0, np.nan]]], [[[-0.5, 90.0, 100.5, np.nan]]]],
        ],
        dtype=dtype,
    )

    # The torch normaliser takes the batch; the numpy one takes a sample at a time
    def with_numpy(method: str, batch: np.ndarray) -> np.ndarray:
        return np.stack([getattr(numpy_normaliser, method)(sample) for sample in batch])

    # `assert_allclose` counts NaN as equal, so this also pins where the missing pixels survive
    normalised = torch_normaliser.normalise(torch.from_numpy(values))
    np.testing.assert_allclose(normalised.numpy(), with_numpy("normalise", values), rtol=1e-6)

    filled = torch_normaliser.fill_missing(normalised)
    np.testing.assert_allclose(
        filled.numpy(), with_numpy("fill_missing", normalised.numpy()), rtol=1e-6
    )
    assert not np.isnan(filled.numpy()).any()

    np.testing.assert_allclose(
        torch_normaliser.denormalise(filled).numpy(),
        with_numpy("denormalise", filled.numpy()),
        rtol=1e-6,
    )
