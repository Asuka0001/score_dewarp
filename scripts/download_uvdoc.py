"""Fetch the official MIT-licensed UVDoc model at a pinned, verified revision."""
import argparse
import hashlib
import json
from pathlib import Path

from download_datasets import curl

REVISION = '4c9b82b537057aff2526e6dd118a847cdd072e82'
# Git blob hashes from the official tree, not hashes supplied by downloaded code.
FILES = {
    'model.py': (11163, '920e561cffa7b42331d6bb2ba27b60a5cd873c16'),
    'LICENSE': (1069, 'a8a3281b35f243bd717b8681b7abe094be2f5718'),
    'model/best_model.pkl': (32158393, '418108b3fcc9b49df64cb3bae0f5506b3e406934'),
}


def verify(path, size, expected):
    data = path.read_bytes()
    actual = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
    if len(data) != size or actual != expected:
        raise ValueError(f'UVDoc source blob mismatch: {path}')
    return hashlib.sha256(data).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path('data/external/uvdoc'))
    args = ap.parse_args()
    records = {}
    for name, (size, blob) in FILES.items():
        path = args.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            part = path.with_suffix(path.suffix + '.part')
            curl(f'https://raw.githubusercontent.com/tanguymagne/UVDoc/{REVISION}/{name}', part)
            verify(part, size, blob)
            part.replace(path)
        records[name] = {'bytes': size, 'git_blob_sha1': blob, 'sha256': verify(path, size, blob)}
    manifest = {'repository': 'https://github.com/tanguymagne/UVDoc',
                'revision': REVISION, 'license': 'MIT', 'files': records}
    (args.root / 'download_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
