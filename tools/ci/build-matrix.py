"""Select a diagnostic build matrix, never publish a partial-platform release."""

import json
import os
from pathlib import Path

PLATFORMS = {
    "macos": [{"os": "macos-15", "target": "aarch64-apple-darwin"}],
    "linux": [
        {"os": "ubuntu-24.04", "target": "x86_64-unknown-linux-gnu"},
        {"os": "ubuntu-24.04-arm", "target": "aarch64-unknown-linux-gnu"},
    ],
    "windows": [{"os": "windows-2022", "target": "x86_64-pc-windows-msvc"}],
}


def select_matrix(platform, build_only):
    if platform != "all" and not build_only:
        raise ValueError("Platform filtering is only allowed with build_only=true")
    if platform == "all":
        return {"include": [row for rows in PLATFORMS.values() for row in rows]}
    if platform not in PLATFORMS:
        raise ValueError(f"Unknown platform: {platform}")
    return {"include": PLATFORMS[platform]}


if __name__ == "__main__":
    matrix = select_matrix(os.environ.get("BUILD_PLATFORM", "all"), os.environ.get("BUILD_ONLY") == "true")
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
        output.write("matrix=" + json.dumps(matrix, separators=(",", ":")) + "\n")
    print(json.dumps(matrix, indent=2))
