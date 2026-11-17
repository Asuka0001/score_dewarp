"""Download a pinned official CPMS release and verify every Git blob.

Photos, labels and semantic crops stay in the git-ignored data directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from download_datasets import curl, extract, get_json

REVISION = '4d091ca8a68d7a7d4a93e24dd4f6a4033b2ca7ca'
REPOSITORY = 'https://github.com/itec-hust/CPMS'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path('data/external/cpms'))
    args = ap.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    tree = get_json(f'https://api.github.com/repos/itec-hust/CPMS/git/trees/{REVISION}?recursive=1', root/'source_tree.json')
    if tree.get('truncated') or tree.get('sha') != REVISION:
        raise ValueError('Unexpected or incomplete source tree')
    archive = root/'cpms.zip'
    part = root/'cpms.zip.part'
    if not archive.exists():
        # A completed manual download can be reused, but must pass ZIP CRC.
        if not part.exists() or not zipfile.is_zipfile(part):
            curl(f'https://codeload.github.com/itec-hust/CPMS/zip/{REVISION}', part, resume=part.exists())
        with zipfile.ZipFile(part) as z:
            if z.testzip() is not None:
                raise ValueError('CPMS archive CRC failed')
        part.replace(archive)
    extract(archive, root/'source')
    source = root/'source'/('CPMS-'+REVISION)
    records = []
    for row in tree['tree']:
        if row['type'] != 'blob':
            continue
        path = source/row['path']
        if not path.resolve().is_relative_to(source.resolve()):
            raise ValueError('Unsafe source path')
        data = path.read_bytes()
        git_hash = hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
        if len(data) != row['size'] or git_hash != row['sha']:
            raise ValueError(f'Source blob mismatch: {path}')
        records.append({'path': row['path'], 'size': len(data), 'git_blob_sha1': git_hash,
                        'sha256': hashlib.sha256(data).hexdigest()})
    images = sorted((source/'data').glob('*.jpeg'))
    labels = sorted((source/'label').glob('*.json'))
    if not images or {p.stem for p in images} != {p.stem for p in labels}:
        raise ValueError('Photo/label pairing mismatch')
    result = {'repository': REPOSITORY, 'revision': REVISION, 'source': str(source),
              'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
              'photos': len(images), 'labels': len(labels), 'verified_blobs': len(records),
              'bytes': sum(r['size'] for r in records), 'files': records}
    (root/'download_manifest.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    main()
