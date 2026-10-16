"""Download the EMIT L1B radiance granules listed in granules.json.

python download.py DATA_DIR

Needs earthaccess and an Earthdata login (~/.netrc or EARTHDATA_USERNAME
and EARTHDATA_PASSWORD).
"""

import json
import sys
from pathlib import Path

import earthaccess

dest = Path(sys.argv[1])
dest.mkdir(parents=True, exist_ok=True)
granules = json.loads((Path(__file__).parent / "granules.json").read_text())

earthaccess.login()
for g in granules:
    if (dest / f"{g['granule']}.nc").exists():
        continue
    results = earthaccess.search_data(short_name="EMITL1BRAD", granule_name=g["granule"])
    links = [u for u in results[0].data_links() if Path(u).name == f"{g['granule']}.nc"]
    earthaccess.download(links, str(dest))
