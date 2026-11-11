"""Download official DeepScoresV2 Dense / Doc3D archives, verify and extract.

Uses system curl (including the system TLS trust store), with resumable .part
files. Doc3D requires accepting the HF dataset conditions and a local HF login.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import zipfile


HF_REPO = "https://huggingface.co/datasets/StonyBrook-CVLab/doc3D-dataset"
HF_API = "https://huggingface.co/api/datasets/StonyBrook-CVLab/doc3D-dataset/tree/main/doc3d"
ZENODO = "https://zenodo.org/api/records/4012193"


def hf_token():
    token_path = Path(os.environ.get("HF_TOKEN_PATH", Path(os.environ.get(
        "HF_HOME", Path.home() / ".cache/huggingface")) / "token"))
    token = os.environ.get("HF_TOKEN")
    if not token and token_path.is_file():
        token = token_path.read_text().strip()
    if token and any(c in token for c in '\r\n"\\'):
        raise ValueError("Invalid HF token format")
    return token


def curl(url, path, token=None, resume=False):
    cmd = ["curl", "-fL", "--silent", "--show-error", "--retry", "3",
           "--connect-timeout", "30", "--speed-time", "60", "--speed-limit", "1024"]
    if resume:
        cmd += ["-C", "-"]
    # Keep credentials out of argv and logs. curl does not forward Authorization
    # to a different redirect host unless --location-trusted is used.
    config = ""
    if token:
        cmd += ["--config", "-"]
        config = f'header = "Authorization: Bearer {token}"\n'
    cmd += ["-o", str(path), "--write-out", "%{http_code}", url]
    result = subprocess.run(cmd, input=config, text=True, stdout=subprocess.PIPE)
    if result.stdout.strip() in ("401", "403") and url.startswith(HF_REPO + "/"):
        raise SystemExit(
            "Doc3D access denied (HTTP " + result.stdout.strip() + "). "
            "A local HF login alone is insufficient: request dataset access at "
            + HF_REPO + " and wait until the account is authorized. "
            "If already approved, check that the local login uses the same account "
            "and that its token permits access to this gated dataset. "
            "Keep any partial downloads; rerun this command after access is granted.")
    if result.returncode:
        raise SystemExit(f"Download failed (curl exit {result.returncode}); "
                         "partial downloads have been kept for retry.")


def get_json(url, dest, token=None):
    if not dest.exists():
        temp = dest.with_suffix(".json.part")
        curl(url, temp, token)
        json.loads(temp.read_text())
        temp.replace(dest)
    return json.loads(dest.read_text())


def verify(path, size, checksum=None):
    if path.stat().st_size != size:
        raise ValueError(f"Size mismatch: {path}")
    if checksum:
        algorithm, expected = checksum.split(":", 1)
        with path.open("rb") as f:
            actual = hashlib.file_digest(f, algorithm).hexdigest()
        if actual != expected:
            raise ValueError(f"Checksum mismatch: {path}; remove the corrupt archive before retrying")


def extract(path, dest):
    marker = dest / ("." + path.name + ".extracted")
    if marker.exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            for member in z.infolist():
                resolved = (dest / member.filename).resolve()
                if not resolved.is_relative_to(dest.resolve()):
                    raise ValueError("Unsafe zip member")
                if member.external_attr >> 16 & 0o170000 == 0o120000:
                    raise ValueError("Zip symlinks are not supported")
            z.extractall(dest)  # zipfile checks each member's CRC during extraction
    else:
        with tarfile.open(path) as t:
            t.extractall(dest, filter="data")
    marker.write_text("verified and extracted\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", choices=["deepscores", "doc3d"])
    ap.add_argument("--root", type=Path, default=Path("data/external"))
    ap.add_argument("--shards", type=int, nargs="+", default=[8], help="Doc3D shards, 1..21; 8 is smallest")
    args = ap.parse_args()
    if any(not 1 <= x <= 21 for x in args.shards):
        ap.error("Doc3D shard IDs must be between 1 and 21")
    archives, metadata = args.root / "archives", args.root / "metadata"
    archives.mkdir(parents=True, exist_ok=True)
    metadata.mkdir(parents=True, exist_ok=True)
    token = hf_token() if args.dataset == "doc3d" else None
    if args.dataset == "deepscores":
        record = get_json(ZENODO, metadata / "deepscores.json")
        files = [(f["key"], f["size"], f["links"]["self"], f["checksum"])
                 for f in record["files"] if f["key"] == "ds2_dense.tar.gz"]
    else:
        listing = get_json(HF_API, metadata / "doc3d-files.json", token)
        wanted = {f"{p}_{s}.zip" for p in ("img", "uv") for s in args.shards}
        files = [(Path(f["path"]).name, f["size"], HF_REPO + "/resolve/main/" + f["path"], None)
                 for f in listing if Path(f["path"]).name in wanted]
        if len(files) != len(wanted):
            raise ValueError("Incomplete Doc3D file listing")
        if not token and any(not (archives / f[0]).exists() for f in files):
            raise SystemExit("Doc3D is gated: accept conditions at " + HF_REPO +
                             " and run `hf auth login` locally (or set HF_TOKEN). Do not share tokens in chat.")
    print(f"Selected archives: {sum(f[1] for f in files) / 1e9:.2f} GB", flush=True)
    for name, size, url, checksum in files:
        path = archives / name
        if not path.exists():
            temp = archives / (name + ".part")
            print("Downloading", name, flush=True)
            if not temp.exists() or temp.stat().st_size != size:
                curl(url, temp, token, resume=True)
            verify(temp, size, checksum)
            temp.replace(path)
        verify(path, size, checksum)
        print("Verified; extracting", name, flush=True)
        extract(path, args.root / args.dataset)
    print("Ready:", args.root / args.dataset)


if __name__ == "__main__":
    main()
