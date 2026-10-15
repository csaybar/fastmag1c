"""Read EMIT L1B radiance granules. Needs h5py: pip install "fastmag1c[emit]"."""

import numpy as np

WINDOW = (2122, 2488)


def read_emit(path, window=WINDOW):
    """Radiance (downtrack, crosstrack, bands) and wavelengths inside window (nm).

    Only the bands in the window are read from disk. The data stay in sensor
    geometry, which is what the column-wise filters expect.
    """
    import h5py

    with h5py.File(path, "r") as f:
        wl = f["sensor_band_parameters/wavelengths"][:]
        idx = np.flatnonzero((wl >= window[0]) & (wl <= window[1]))
        rad = f["radiance"][:, :, idx[0]:idx[-1] + 1]
    return rad, wl[idx]


def orthorectify(image, path, nodata=-9999):
    """Map a (downtrack, crosstrack) array to the granule's lat/lon grid with the GLT.

    Returns the gridded array, its GDAL geotransform
    (x0, dx, 0, y0, 0, dy) and the CRS as WKT, both read from the granule.
    """
    import h5py

    with h5py.File(path, "r") as f:
        gx = f["location/glt_x"][:].astype(np.int64)
        gy = f["location/glt_y"][:].astype(np.int64)
        transform = tuple(float(v) for v in f.attrs["geotransform"])
        crs = f.attrs["spatial_ref"].decode()
    out = np.full(gx.shape, nodata, dtype=image.dtype)
    ok = (gx > 0) & (gy > 0)
    out[ok] = image[gy[ok] - 1, gx[ok] - 1]
    return out, transform, crs
