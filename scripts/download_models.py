"""Restore the learned staff segmentation model (homr segnet, ONNX) if missing.

    python scripts/download_models.py

The weights ship with this repository in models/homr/ (AGPL-3.0, from homr,
https://github.com/liebharc/homr; trained on DeepScoresV2, CC BY 4.0, see
models/homr/NOTICE.md). This script only re-downloads them from the homr
release, e.g. after a shallow / LFS-less checkout. Used by ``--seg homr`` /
``MetricConfig(seg="homr")``.
"""
import hashlib
import os
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scoredewarp.core.learned import DEFAULT_PATH, SEGNET_URL  # noqa: E402


def main():
    dst = DEFAULT_PATH
    if dst.exists():
        print("already present:", dst)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    z = dst.with_suffix(".zip")
    subprocess.run(["curl", "-fL", "--retry", "3", "-o", str(z), SEGNET_URL], check=True)
    with zipfile.ZipFile(z) as f:
        f.extract(dst.name, dst.parent)
    z.unlink()
    print("saved", dst, f"{dst.stat().st_size / 1e6:.1f} MB",
          "sha256", hashlib.sha256(dst.read_bytes()).hexdigest()[:16])


if __name__ == "__main__":
    main()
