import nd2
import os
import pathlib

import xarray as xr
from typing import Literal

import numpy as np
import numpy.typing as npt

from ._canonicalize_video import canonicalize_video

def parse_video(
        path: str | os.PathLike,
        T: int | None = None,
        C: int | None = None,
        Z: int | None = None,
        Y: int | None = None,
        X: int | None = None,
        dt: float | None = None,
        dz: float | None = None,
        dy: float | None = None,
        dx: float | None = None,
        dtype: npt.DTypeLike | Literal["uint12"] | None = None,
    ) -> xr.DataArray:

    # Determine scheme and suffix
    if isinstance(path, str):
        suffix = pathlib.Path(path).suffix

    # Gather all keyword arguments for those load functions that require them.
    kwargs = {
        "T": T,
        "C": C,
        "Z": Z,
        "Y": Y,
        "X": X,
        "dt": dt,
        "dz": dz,
        "dy": dy,
        "dx": dx,
        "dtype": dtype,
    }

    # Load the path.
    #protocols = fsspec.available_protocols()
    match ("file", suffix):
        case (_, ".nd2"):
            #videof = nd2.ND2File(path)
            image = nd2.imread(path, xarray=True, dask=True)

    return canonicalize_video(image)