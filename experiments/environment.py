"""Write the hardware and software used for the experiments to environment.txt.

python environment.py
"""

import platform
import subprocess
from pathlib import Path

import torch

import fastmag1c


def run(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


lines = [
    f"fastmag1c {fastmag1c.__version__}",
    f"torch {torch.__version__}, CUDA {torch.version.cuda}",
    f"python {platform.python_version()}",
    f"cpu {run('lscpu | grep -iE \"model name|nombre del modelo\" | cut -d: -f2 | xargs')}",
    f"cpu threads used by torch {torch.get_num_threads()}",
    f"blas {'MKL' if torch.backends.mkl.is_available() else 'not MKL'}",
]
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        lines.append(f"gpu {i} {p.name}, {p.total_memory / 2**30:.0f} GiB")
    lines.append(f"driver {run('nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1')}")
lines.append("tf32 disabled for float32 matrix products")

(Path(__file__).parent / "environment.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
