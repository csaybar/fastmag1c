"""Run MAG1C and FastMAG1C on the same scene and compare time and output."""

import time
from dataclasses import dataclass

import numpy as np
import torch

from .scene import NODATA, retrieve
from .template import load_template


@dataclass
class Comparison:
    mag1c: np.ndarray
    fastmag1c: np.ndarray
    time_mag1c: float  # mean over repeats, in seconds
    time_fastmag1c: float
    repeats: int
    threshold: float
    device: str
    dtype: str
    reference: np.ndarray | None = None  # MAG1C in float64

    @property
    def speedup(self):
        return self.time_mag1c / self.time_fastmag1c

    def errors(self, a="fastmag1c", b="mag1c", plume=False):
        """max, mean and P99 of |a - b| in ppm·m, where a and b are
        "mag1c", "fastmag1c" or "reference". With plume=True only pixels
        above the threshold in the reference (or MAG1C) map are used."""
        base = self.mag1c if self.reference is None else self.reference
        valid = base != NODATA
        if plume:
            valid &= base > self.threshold
        diff = np.abs(getattr(self, a)[valid] - getattr(self, b)[valid])
        if diff.size == 0:
            return {"n": 0, "max": np.nan, "mean": np.nan, "p99": np.nan}
        return {"n": int(diff.size), "max": float(diff.max()),
                "mean": float(diff.mean()), "p99": float(np.percentile(diff, 99))}

    def __repr__(self):
        pairs = [("FastMAG1C - MAG1C", "fastmag1c", "mag1c")]
        if self.reference is not None:
            pairs += [("MAG1C - MAG1C f64", "mag1c", "reference"),
                      ("FastMAG1C - MAG1C f64", "fastmag1c", "reference")]
        n_plume = self.errors(plume=True)["n"]
        lines = [
            f"{'':<24}{'MAG1C':>10}{'FastMAG1C':>12}",
            f"{'time (s)':<24}{self.time_mag1c:>10.2f}{self.time_fastmag1c:>12.2f}",
            f"speedup {self.speedup:.2f}x on {self.device}, {self.dtype}, "
            f"mean of {self.repeats} run{'s' if self.repeats > 1 else ''}",
        ]
        for name, plume in [("all pixels", False),
                            (f"{n_plume} plume pixels (> {self.threshold:g} ppm·m)", True)]:
            lines += ["", name, f"{'|difference| ppm·m':<24}{'max':>12}{'mean':>12}{'p99':>12}"]
            for label, a, b in pairs:
                e = self.errors(a, b, plume)
                lines.append(f"{label:<24}{e['max']:>12.2e}{e['mean']:>12.2e}{e['p99']:>12.2e}")
        return "\n".join(lines)


def _timed(rad, method, device, repeats, **kwargs):
    # warm-up on a few columns so CUDA and BLAS start-up costs are not timed
    retrieve(rad[:, :8], method=method, device=device, **kwargs)
    times = []
    for _ in range(repeats):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        mf = retrieve(rad, method=method, device=device, **kwargs)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return mf, float(np.mean(times))


def compare(rad, wavelengths=None, *, template=None, device=None,
            dtype=torch.float32, threshold=500.0, reference=False, repeats=3,
            **kwargs):
    """Retrieve with MAG1C and FastMAG1C and report runtime and differences.

    Each method runs once on a few columns to warm up and then repeats times
    on the full scene, and the mean time is reported. threshold (ppm·m)
    selects the plume pixels. With reference=True, MAG1C is also run in
    float64 so the float32 error of each method can be measured against it.
    Other keyword arguments go to retrieve().
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    if template is None:
        template = load_template(wavelengths)

    kwargs["template"] = template
    ref, t_ref = _timed(rad, "mag1c", device, repeats, dtype=dtype, **kwargs)
    new, t_new = _timed(rad, "fastmag1c", device, repeats, dtype=dtype, **kwargs)
    f64 = None
    if reference:
        f64 = retrieve(rad, method="mag1c", device=device, dtype=torch.float64, **kwargs)
    return Comparison(ref, new, t_ref, t_new, repeats, threshold, device,
                      str(dtype).removeprefix("torch."), f64)
