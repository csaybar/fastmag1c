"""Run a filter over a full scene, one CCD column at a time."""

import torch

from . import filters
from .template import load_template

NODATA = -9999

METHODS = {
    "rmf": filters.rmf,
    "mag1c": filters.mag1c,
    "fastmag1c": filters.fastmag1c,
}


def retrieve(rad, wavelengths=None, *, template=None, method="fastmag1c",
             device=None, dtype=torch.float32, batch_size=64, **kwargs):
    """CH4 enhancement map in ppm·m, as a numpy array.

    rad is (downtrack, crosstrack, bands) in sensor geometry. Each crosstrack
    column keeps its own statistics, and batch_size columns run together.
    Pixels with NODATA in any band are left out of the column statistics
    and come back as NODATA, as do columns with fewer valid pixels than bands.

    Pass either the band wavelengths in nm (EMIT template) or your own template.
    Extra keyword arguments go to the filter (num_iter, alpha, gamma).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if template is None:
        template = load_template(wavelengths)

    fn = METHODS[method]
    t = torch.as_tensor(template, device=device, dtype=dtype)
    rad = torch.as_tensor(rad).to(device=device, dtype=dtype)
    D, C, s = rad.shape

    # NODATA pixels are filled with the column mean and left out of the
    # column statistics through the mask w
    nodata = (rad == NODATA).any(dim=-1)
    x = rad.masked_fill(nodata[..., None], 0.0)
    count = (~nodata).sum(dim=0)
    mean = x.sum(dim=0) / count.clamp(min=1)[:, None]
    x = torch.where(nodata[..., None], mean, x).permute(1, 0, 2)
    w = (~nodata).T[..., None].to(dtype)

    # columns with fewer valid pixels than bands have a singular covariance
    cols = torch.nonzero(count > s).squeeze(1)
    out = torch.full((C, D), NODATA, device=device, dtype=dtype)
    for i in range(0, len(cols), batch_size):
        c = cols[i:i + batch_size]
        wc = w[c] if nodata[:, c].any() else None
        mf, _ = fn(x[c], t, w=wc, **kwargs)
        out[c] = mf[..., 0]

    out = out.T
    out[nodata] = NODATA
    return out.cpu().numpy()
