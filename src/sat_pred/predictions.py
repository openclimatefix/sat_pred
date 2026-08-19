"""The layout the prediction stores are written in

The backtest and the production inference app both write forecasts made by the same models, and
`cloudcasting_metrics` scores one store against the other. That alignment only holds if the two
name their data variable the same thing and lay their dimensions out in the same order, so both
build their coordinates from here rather than each spelling them out.
"""

import pandas as pd
import xarray as xr
from numpy.typing import NDArray

PREDICTION_VAR_NAME = "sat_pred"

# Init-time leads, so a store can be written an init-time at a time - the app writes a single
# forecast, and the backtest writes each batch into its own block of a store laid out up front
PREDICTION_DIMS = ("init_time_utc", "channel", "step", "y_geostationary", "x_geostationary")


def forecast_steps(forecast_mins: int, sample_freq_mins: int) -> pd.TimedeltaIndex:
    """The lead times a model predicts for

    These start one step after the init-time rather than at zero. The models predict the frames
    which follow t0; the frame at t0 itself is an input.

    Args:
        forecast_mins: How far ahead the model predicts, in minutes
        sample_freq_mins: The spacing between the frames the model was trained on, in minutes
    """
    return pd.timedelta_range(
        start=f"{sample_freq_mins}min",
        end=f"{forecast_mins}min",
        freq=f"{sample_freq_mins}min",
    )


def prediction_coords(
    da: xr.DataArray,
    forecast_mins: int,
    sample_freq_mins: int,
) -> dict[str, NDArray | pd.TimedeltaIndex]:
    """The coordinates every prediction shares, in `PREDICTION_DIMS` order

    `init_time_utc` is left out, because it is the one coordinate the two writers legitimately
    differ on - the app writes a single init-time and the backtest writes every init-time of the
    period it covers. Both put it first, so a caller builds its own coords as
    `{"init_time_utc": ..., **prediction_coords(...)}`.

    Args:
        da: The satellite data the model predicts from. Its channel and spatial coordinates carry
            through to the predictions unchanged - the model predicts the same channels over the
            same grid it was given
        forecast_mins: How far ahead the model predicts, in minutes
        sample_freq_mins: The spacing between the frames the model was trained on, in minutes
    """
    return {
        "channel": da.channel.values,
        "step": forecast_steps(forecast_mins, sample_freq_mins),
        "y_geostationary": da.y_geostationary.values,
        "x_geostationary": da.x_geostationary.values,
    }
