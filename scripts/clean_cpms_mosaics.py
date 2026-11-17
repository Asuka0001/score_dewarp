"""Regenerate CPMS mosaics without patchwise lighting seams; preserve originals."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path

from prepare_diverse_training import generate


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=Path('data/prepared/diverse_20261115'))
    ap.add_argument('--out',type=Path,default=Path('data/prepared/diverse_clean_20261115'))
    ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()):ap.error('Use a new empty directory')
    args.out.mkdir(parents=True)
    source=json.loads((args.source/'manifest.json').read_text())
    jobs=[j for j in json.loads((args.source/'generation_jobs.json').read_text()) if j['type']=='cpms']
    for j in jobs:j['destination']=str(args.out/j['split']/Path(j['destination']).name)
    protocol={'source_manifest_sha256':hashlib.sha256((args.source/'manifest.json').read_bytes()).hexdigest(),
              'changes':'normalize each teacher strip illumination before page composition; reapply continuous photo lighting in simulator',
              'geometry_seeds_and_splits_unchanged':True,'cpms_test_used':False,
              'generator_sha256':hashlib.sha256(Path('scripts/prepare_diverse_training.py').read_bytes()).hexdigest()}
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    replacements=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for row in pool.map(generate,jobs):
            replacements.append(row)
            if len(replacements)%100==0:print('clean mosaics',len(replacements),'/',len(jobs),flush=True)
    rows=[r for r in source['samples'] if r['source']!='cpms_training_texture']+replacements
    # Preserve the original record order to make cache provenance easy to audit.
    by_name={Path(r['path']).name:r for r in rows}
    rows=[by_name[Path(r['path']).name] for r in source['samples']]
    source['samples']=rows
    (args.out/'manifest.json').write_text(json.dumps(source,indent=2))
    print('cleaned',len(replacements),'paired samples',flush=True)


if __name__=='__main__':main()
