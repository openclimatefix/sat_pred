"""The spatial grid a model was trained on, and selecting model inputs onto it

The training data is a crop of the full satellite disc - for the UK models, 614 x 372 pixels at
about 3km. Nothing about that crop is recorded in the model config, and the models are fully
convolutional, so one handed a different crop at inference time runs happily and predicts plausible
values over the wrong geography. The grid is therefore recorded alongside the model when training
starts, and the inputs are selected onto it before they reach the model: an archive covering more
than the model needs is cropped to it, and one which cannot cover it is refused.

The coordinates are recorded as full arrays rather than as an extent and a size. The grid is
regular, so an extent would determine it, but rebuilding it with `linspace` would not reproduce the
original floats bit for bit, and the selection is an exact lookup.
"""

from pathlib import Path

import numpy as np
import xarray as xr
from numpy.typing import NDArray


class SpatialGrid:
    """The spatial grid a model was trained on"""

    def __init__(
        self,
        x_geostationary: NDArray[np.float64],
        y_geostationary: NDArray[np.float64],
    ) -> None:
        """The spatial grid a model was trained on

        Both coordinates must be as `open_sat_data` leaves them. It runs
        `make_spatial_coords_increasing`, which reverses `x_geostationary` - the stores hold it
        descending - so a grid recorded on one side of that transform would never match one
        recorded on the other.

        Args:
            x_geostationary: The x coordinate of every pixel column
            y_geostationary: The y coordinate of every pixel row
        """
        self.x_geostationary = x_geostationary
        self.y_geostationary = y_geostationary

    @classmethod
    def from_dataarray(cls, da: xr.DataArray) -> "SpatialGrid":
        """The grid some satellite data is on

        Args:
            da: Satellite data as `open_sat_data` returns it
        """
        return cls(da.x_geostationary.values, da.y_geostationary.values)

    @classmethod
    def load(cls, path: str | Path) -> "SpatialGrid":
        """Load a grid saved by `save`

        Args:
            path: Path of the file to load
        """
        with np.load(path) as file:
            return cls(file["x_geostationary"], file["y_geostationary"])

    def save(self, path: str | Path) -> None:
        """Save the grid so that `load` can read it back

        Numpy's own container rather than one of the YAML configs saved beside it, because two
        float64 coordinate arrays round-trip through it exactly, without the check having to depend
        on how a text format happens to render a float.

        Args:
            path: Path of the file to write. Numpy appends `.npz` if it is not already there
        """
        np.savez(path, x_geostationary=self.x_geostationary, y_geostationary=self.y_geostationary)

    def select(self, da: xr.DataArray) -> xr.DataArray:
        """Select the pixels of this grid out of some satellite data

        Data covering more than the grid is cropped to it. Data which does not cover all of it
        raises, because a model cannot be run over an area it was never trained on.

        Args:
            da: Satellite data as `open_sat_data` returns it
        """
        return da.sel(x_geostationary=self.x_geostationary, y_geostationary=self.y_geostationary)
