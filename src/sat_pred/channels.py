"""The satellite channels a model is trained on, and how each one is normalised

A `ChannelConfig` is a single ordered mapping of channel name to normalisation constants. It is the
one place which decides both which channels a model sees and what order they arrive in, so the
channel a model was trained to read at index `n` is always the one the config lists `n`th. The
samples are selected out of the store in this order rather than the order the store happens to hold
them in - see `SatelliteDataset`.

Each channel is clipped to a plausible physical range and then converted to a z-score:

    z = (clip(x, clip_min, clip_max) - mean) / std

The satellite stores hold physical units - reflectance percentages for the visible and near infrared
channels, and brightness temperatures in Kelvin for the rest - so the channels cover wildly
different ranges without this. The clip also removes the spurious values which occasionally appear
in the archive and would otherwise dominate the loss.

Missing pixels stay NaN through normalisation so they can be masked out of the loss - see
`TrainingModule._calculate_common_losses`. The model cannot consume NaNs, so they are replaced in
the model input only, by `ChannelNormaliser.fill_missing()`, with the channel's `missing_value`.

See `configs/datamodule/channels/` for the configs the training runs use.
"""

from collections.abc import Mapping, Sequence
from functools import cached_property
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import torch
import yaml
from numpy.typing import NDArray
from pydantic import BaseModel, Field, RootModel, model_validator

class ChannelNormalisation(BaseModel):
    """How a single satellite channel is clipped and z-scored"""

    mean: float = Field(description="Mean of the channel after clipping, in physical units")
    std: float = Field(gt=0, description="Standard deviation of the channel after clipping")
    clip_min: float = Field(description="Values below this are clipped up to it")
    clip_max: float = Field(description="Values above this are clipped down to it")
    missing_value: float = Field(
        description=(
            "What the model is shown in place of a missing pixel, in physical units. Usually set "
            "below `clip_min` so it cannot be confused for a real reading, but that is not "
            "enforced - setting it to `mean` to fill gaps with the channel average is equally valid"
        )
    )

    @model_validator(mode="after")
    def _check_clip_range(self) -> "ChannelNormalisation":
        if self.clip_min >= self.clip_max:
            raise ValueError(f"clip_min ({self.clip_min}) must be below clip_max ({self.clip_max})")
        return self


class ChannelConfig(RootModel[dict[str, ChannelNormalisation]]):
    """The channels a model is trained on, in order, and how each one is normalised

    The order the channels are written in the config is the order they appear in the samples, so
    reordering an existing config changes what a model trained against it sees at each index.
    """

    @model_validator(mode="after")
    def _check_not_empty(self) -> "ChannelConfig":
        if not self.root:
            raise ValueError("A channel config must name at least one channel")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ChannelConfig":
        """Load a channel config from a YAML file"""
        with open(path) as file:
            return cls.model_validate(yaml.safe_load(file))

    def __getitem__(self, channel: str) -> ChannelNormalisation:
        if channel not in self.root:
            raise KeyError(
                f"Channel {channel!r} is not in the channel config. It holds: {list(self.root)}"
            )
        return self.root[channel]

    def __len__(self) -> int:
        return len(self.root)

    @property
    def names(self) -> list[str]:
        """The channel names, in the order they appear in the samples"""
        return list(self.root)

    @cached_property
    def normaliser(self) -> "ChannelNormaliser":
        """The normaliser which applies these constants to a sample"""
        return ChannelNormaliser(list(self.root.values()))


class ChannelNormaliser:
    """Normalises samples shaped (channel, time, y, x) for a fixed, ordered set of channels"""

    def __init__(self, normalisations: Sequence[ChannelNormalisation]):
        """Normalises samples shaped (channel, time, y, x)

        Args:
            normalisations: The constants of each channel, in the order the channels appear in the
                samples
        """
        if len(normalisations) == 0:
            raise ValueError("Need the constants of at least one channel")

        def column(attribute: str) -> NDArray[np.float32]:
            # Shaped to broadcast against the (channel, time, y, x) samples
            column = [getattr(normalisation, attribute) for normalisation in normalisations]
            return np.array(column, dtype=np.float32).reshape(-1, 1, 1, 1)

        self.mean = column("mean")
        self.std = column("std")
        self.clip_min = column("clip_min")
        self.clip_max = column("clip_max")
        self.missing_value = column("missing_value")

    @property
    def missing_fill_value(self) -> NDArray[np.float32]:
        """The normalised value which stands in for missing pixels in the model input

        The configured `missing_value` is z-scored but deliberately not clipped - clipping it would
        pull it back to `clip_min` and it could never sit outside the range of the real data.
        """
        return (self.missing_value - self.mean) / self.std

    def normalise(self, values: NDArray) -> NDArray[np.float32]:
        """Clip each channel to its physical range and convert it to a z-score

        Missing pixels stay NaN.
        """
        clipped = np.clip(values, self.clip_min, self.clip_max)
        return ((clipped - self.mean) / self.std).astype(np.float32)

    def denormalise(self, values: NDArray) -> NDArray[np.float32]:
        """Convert z-scores back to physical units"""
        return (values * self.std + self.mean).astype(np.float32)

    def fill_missing(self, values: NDArray) -> NDArray[np.float32]:
        """Replace missing pixels with `missing_fill_value`

        Convolutions spread a NaN across everything downstream of it, so the model input cannot
        carry missing pixels as NaN the way the target does.
        """
        return np.where(np.isnan(values), self.missing_fill_value, values).astype(np.float32)


class TorchChannelNormaliser:
    """A `ChannelNormaliser`'s constants, held as tensors on the device the model runs on

    This normalises **batched** samples shaped (batch, channel, time, y, x), whereas
    `ChannelNormaliser` works on a single (channel, time, y, x) sample - that leading dimension is
    why the constants here carry one more axis than the numpy ones.

    Normalising on the model's device rather than in the dataloader workers means the samples reach
    the main process as the raw values the store holds, which for a float16 store is half the bytes
    of the normalised float32 the workers would otherwise send, and puts none of the arithmetic on
    the process the model is waiting on.

    The interface and the arithmetic are the same as `ChannelNormaliser`'s, in the same float32,
    so the two give the same predictions - `tests/test_channels.py` pins them together.
    """

    def __init__(self, normaliser: ChannelNormaliser, device: torch.device):
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
        """Clip each channel to its physical range and convert it to a z-score

        Missing pixels stay NaN.

        Samples arrive in whatever dtype the store holds and are cast to float32 up front, so the
        arithmetic runs at a fixed precision rather than one the input dtype picks by promotion.
        Note this narrows float64 - the one input for which `ChannelNormaliser`, which casts at the
        end instead, gives a different answer. The stores hold float32 or float16.
        """
        clipped = values.to(torch.float32).clamp(self.clip_min, self.clip_max)
        return (clipped - self.mean) / self.std

    def denormalise(self, values: torch.Tensor) -> torch.Tensor:
        """Convert z-scores back to physical units"""
        return values * self.std + self.mean

    def fill_missing(self, values: torch.Tensor) -> torch.Tensor:
        """Replace missing pixels with `missing_fill_value`

        Convolutions spread a NaN across everything downstream of it, so the model input cannot
        carry missing pixels as NaN the way the target does.
        """
        return torch.where(values.isnan(), self.missing_fill_value, values)


# The forms a channel config can be supplied in. Hydra passes a mapping read from the config
ChannelConfigInput: TypeAlias = ChannelConfig | Mapping[str, Any] | str | Path


def parse_channel_config(channels: ChannelConfigInput) -> ChannelConfig:
    """Accept a channel config in any of the forms a caller might supply it

    Args:
        channels: An already-parsed config, a mapping of channel name to normalisation constants -
            which is what hydra passes in - or the path of a YAML file holding that mapping
    """
    if isinstance(channels, ChannelConfig):
        return channels
    if isinstance(channels, str | Path):
        return ChannelConfig.from_yaml(channels)
    if isinstance(channels, Mapping):
        # Hydra supplies a DictConfig, which pydantic cannot read the nested values out of
        return ChannelConfig.model_validate(
            {channel: dict(normalisation) for channel, normalisation in channels.items()}
        )
    raise TypeError(
        f"Cannot read a channel config from {type(channels).__name__}. Expected a mapping of "
        "channel name to normalisation constants, or the path of a YAML file holding one"
    )
