"""CH4 unit absorption template for EMIT.

MODTRAN look-up table convolved with the EMIT spectral response functions,
sampled at the 285 EMIT band centers.
"""

import urllib.request
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

URL = "https://data.source.coop/taco/methaneset/ch4_emit.safetensors"
CACHE = Path.home() / ".cache" / "fastmag1c" / "ch4_emit.safetensors"


def load_template(wavelengths, path=CACHE):
    """CH4 template at the given band centers (nm), interpolated linearly."""
    path = Path(path)
    if not path.exists():
        # download to a temporary name so an interrupted download is not cached
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        req = urllib.request.Request(URL, headers={"User-Agent": "fastmag1c"})
        with urllib.request.urlopen(req) as r:
            tmp.write_bytes(r.read())
        tmp.replace(path)

    tensors = load_file(str(path))
    centers, template = tensors["centers"], tensors["template"]
    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    if wavelengths.min() < centers[0] or wavelengths.max() > centers[-1]:
        raise ValueError(
            f"wavelengths must lie within {centers[0]:.1f}-{centers[-1]:.1f} nm, "
            f"the range of the EMIT template; pass template= for other sensors"
        )
    return np.interp(wavelengths, centers, template)
