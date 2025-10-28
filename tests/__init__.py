import json
from pathlib import Path

import scmultim

PACKAGE_DIR = Path(scmultim.__file__).parent
MANIFEST_FILE = PACKAGE_DIR / "__FRACTAL_MANIFEST__.json"
with MANIFEST_FILE.open("r") as f:
    MANIFEST = json.load(f)
