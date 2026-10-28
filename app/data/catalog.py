"""Medical test catalog loader.

NOTE: The critical threshold values defined in test_catalog.json are illustrative
demo thresholds and must be reviewed by a qualified clinician before any real use.
"""

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

CATALOG_PATH = Path(__file__).resolve().parent / "test_catalog.json"


@lru_cache(maxsize=1)
def load_test_catalog() -> list[dict[str, Any]]:
    """Load the test catalog JSON relative to the package location.

    NOTE: The critical threshold values defined in test_catalog.json are illustrative
    demo thresholds and must be reviewed by a qualified clinician before any real use.
    """
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
