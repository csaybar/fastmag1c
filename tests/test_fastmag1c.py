import numpy as np
import pytest
import torch

from fastmag1c import compare, retrieve
from fastmag1c.filters import fastmag1c, mag1c
from fastmag1c.scene import NODATA


def columns(B=4, N=512, s=50, seed=0, dtype=torch.float64):
    # synthetic columns as in the paper: random SPD covariance, positive radiance
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(s, s, generator=g, dtype=torch.float64)
    L = torch.linalg.cholesky(A @ A.T + 1e-3 * torch.eye(s, dtype=torch.float64))
    x = torch.randn(B, N, s, generator=g, dtype=torch.float64) @ L.T
    x = x - x.amin() + 1.0
    t = torch.full((s,), 1 / s**0.5, dtype=torch.float64)
    return x.to(dtype), t.to(dtype)


def plume(x, t, amount=5e-2):
    # add a CH4 signal to a block of pixels so the iterations have work to do
    x = x.clone()
    x[:, :50] += amount * t * x.mean(dim=1, keepdim=True)
    return x


def test_exact_in_float64():
    x, t = columns()
    x = plume(x, t)
    ref, _ = mag1c(x, t)
    new, _ = fastmag1c(x, t)
    assert ref[:, :50].max() > 0
    assert (new - ref).abs().max() < 1e-6


def test_float32_error_against_float64():
    x, t = columns()
    x = plume(x, t)
    ref, _ = mag1c(x, t)
    new, _ = fastmag1c(x.float(), t.float())
    err_new = (new.double() - ref).abs()
    assert err_new.quantile(0.99) < 2.0


@pytest.mark.parametrize("method", ["mag1c", "fastmag1c"])
def test_retrieve_keeps_nodata(method):
    x, t = columns(B=6, N=300, s=20)
    rad = x.permute(1, 0, 2).numpy().copy()
    rad[10, 2, 5] = NODATA
    rad[:, 4] = NODATA
    mf = retrieve(rad, template=t.numpy(), method=method, device="cpu",
                  dtype=torch.float64, batch_size=4)
    assert mf.shape == (300, 6)
    assert mf[10, 2] == NODATA
    assert (mf[:, 4] == NODATA).all()
    assert (mf[mf != NODATA] >= 0).all()


@pytest.mark.parametrize("method", ["mag1c", "fastmag1c"])
def test_nodata_left_out_of_statistics(method):
    # a column with NODATA pixels gives the same result as the column without them
    x, t = columns(B=1, N=300, s=20)
    x = plume(x, t)
    rad = x.permute(1, 0, 2).numpy().copy()
    rad[[3, 77, 150], 0, 2] = NODATA
    keep = np.setdiff1d(np.arange(300), [3, 77, 150])
    kw = dict(template=t.numpy(), method=method, device="cpu", dtype=torch.float64)
    full = retrieve(rad, **kw)
    dropped = retrieve(rad[keep], **kw)
    assert np.abs(full[keep] - dropped).max() < 1e-6


def test_compare():
    x, t = columns(B=8, N=512)
    rad = plume(x, t).permute(1, 0, 2).numpy()
    c = compare(rad, template=t.numpy(), device="cpu", dtype="float32",
                num_iter=10, threshold=100, reference=True)
    assert c.errors("fastmag1c", "reference")["p99"] < 2.0
    assert c.errors(plume=True)["n"] > 0
    assert c.speedup > 0
    assert "speedup" in repr(c)
