#!/usr/bin/env python3
"""Download the two pinned BIG-Bench Hard task families used for screening."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path


COMMIT = "9ee07bd481feebf959a6b59d61ea57bdcf30964d"
BASE_URL = f"https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/{COMMIT}/bbh"
FILES = (
    "logical_deduction_three_objects.json",
    "logical_deduction_five_objects.json",
    "logical_deduction_seven_objects.json",
    "tracking_shuffled_objects_three_objects.json",
    "tracking_shuffled_objects_five_objects.json",
    "tracking_shuffled_objects_seven_objects.json",
)
DESTINATION = Path(__file__).parent / "data" / "bbh"


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    manifest = {"upstream_commit": COMMIT, "files": {}}
    for filename in FILES:
        url = f"{BASE_URL}/{filename}"
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read()
        parsed = json.loads(content)
        if not isinstance(parsed.get("examples"), list):
            raise ValueError(f"Unexpected BBH format in {filename}")
        (DESTINATION / filename).write_bytes(content)
        manifest["files"][filename] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "examples": len(parsed["examples"]),
        }
        print(f"Downloaded {filename}: {len(parsed['examples'])} examples")

    (DESTINATION / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
