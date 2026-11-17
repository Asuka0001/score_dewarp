"""Merge all eligible training sources, reject bad maps, cache model targets."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scoredewarp.learn.dataset import build_targets


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cache_one(job):
    cv2.setNumThreads(1)
    row,destination,size=job
    row=dict(row);folder=Path(row['path']);gt=json.loads((folder/'gt.json').read_text())
    mapping=np.load(folder/'gt_bmap_s2.npy').astype(np.float64)[::4,::4]
    valid=np.isfinite(mapping).all(-1)&(mapping>=0).all(-1)&(mapping<gt['meta']['size']).all(-1)
    cells=valid[:-1,:-1]&valid[1:,:-1]&valid[:-1,1:]&valid[1:,1:]
    dx=mapping[:-1,1:]-mapping[:-1,:-1];dy=mapping[1:,:-1]-mapping[:-1,:-1]
    det=dx[...,0]*dy[...,1]-dx[...,1]*dy[...,0]
    rate=float((det[cells]<=0).mean()) if cells.any() else 1.
    row['coarse_gt_nonpositive_fraction']=rate
    if rate>.001:
        return {**row,'accepted':False,'reason':'coarse GT nonpositive Jacobian rate >0.1%'}
    target=build_targets(str(folder),tuple(size))
    if not all(np.isfinite(v).all() for v in target.values()):raise ValueError(f'Nonfinite target {folder}')
    destination=Path(destination)
    np.savez_compressed(destination,image=np.rint(target['image']*255).astype(np.uint8),fwd=target['fwd'],
                        page=target['page'].astype(np.uint8),inst_h=target['inst_h'].astype(np.uint16),
                        inst_v=target['inst_v'].astype(np.uint16))
    return {**row,'accepted':True,'cache':str(destination),'gt_sha256':sha(folder/'gt.json'),
            'photo_sha256':sha(folder/'photo.jpg'),'map_sha256':sha(folder/'gt_bmap_s2.npy'),
            'cache_sha256':sha(destination),'page_fraction':float(target['page'].mean())}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--extra',type=Path,default=Path('data/prepared/diverse_20261115'))
    ap.add_argument('--out',type=Path,default=Path('data/prepared/diverse_cache_20261115'))
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--reuse-cache',type=Path,help='Verify and hardlink unchanged cached records from a completed cache')
    ap.add_argument('--size',type=int,nargs=2,default=[384,512])
    args=ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()):ap.error('Use a fresh empty cache directory')
    args.out.mkdir(parents=True)
    rows=[];manifests=[]
    for dataset,family in [('deepscores_full_v3','deepscores'),('doc3d_pilot','doc3d'),('structure_suite_20261115_v2','synthetic')]:
        root=ROOT/'data/prepared'/dataset;manifest=root/'manifest.json';manifests.append(manifest)
        for r in json.loads(manifest.read_text())['samples']:
            split=r['path'].split('/')[0]
            if split=='test':continue
            meta=json.loads((root/r['path']/'gt.json').read_text())['meta']
            rows.append({'path':str(root/r['path']),'split':split,'family':family,'source':meta['source'],
                         'source_group':meta['source_group'],'kind':meta['kind'],'profile':meta.get('profile','original')})
    manifest=args.extra/'manifest.json';manifests.append(manifest)
    for r in json.loads(manifest.read_text())['samples']:
        family={'deepscores':'deepscores','cpms_training_texture':'cpms_texture','diverse_synth':'synthetic'}[r['source']]
        rows.append({**{k:r[k] for k in ('path','split','source','source_group','kind','profile')},'family':family})
    # All source variants must remain on one side, including old/new DS views.
    splits={}
    for r in rows:
        key=(r['source'],r['source_group'])
        if key in splits and splits[key]!=r['split']:raise ValueError(f'Source leakage: {key}')
        splits[key]=r['split']
    protocol={'size':args.size,'manifests':{str(p):sha(p) for p in manifests},
              'source_group_overlap':0,'gt_fold_rejection':'stride-4 source map cells, >0.1% nonpositive',
              'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'scoredewarp/learn/dataset.py']}}
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    reuse={}
    if args.reuse_cache:
        previous=json.loads((args.reuse_cache/'complete.json').read_text())
        if previous['size']!=args.size:raise ValueError('Reuse cache resolution differs')
        reuse={r['path']:r for r in previous['records']}
    jobs=[];records=[]
    for i,r in enumerate(rows):
        destination=args.out/(f'{i:06d}.npz');old=reuse.get(r['path'])
        if old:
            folder=Path(r['path'])
            if any(sha(folder/name)!=old[key] for name,key in [('gt.json','gt_sha256'),('photo.jpg','photo_sha256'),('gt_bmap_s2.npy','map_sha256')]):
                raise ValueError('Cannot reuse changed source')
            if sha(old['cache'])!=old['cache_sha256']:raise ValueError('Cannot reuse changed cache')
            import os
            os.link(old['cache'],destination)
            records.append({**old,**r,'cache':str(destination),'reused_from':old['cache']})
        else:jobs.append((r,str(destination),args.size))
    print('verified reusable',len(records),'new targets',len(jobs),flush=True)
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for record in pool.map(cache_one,jobs):
            records.append(record)
            if len(records)%250==0:print('cache',len(records),'/',len(rows),flush=True)
    kept=[r for r in records if r['accepted']]
    result={'size':args.size,'records':kept,'rejected':[r for r in records if not r['accepted']],
            'counts':dict(Counter(r['split'] for r in kept)),
            'family_counts':{s:dict(Counter(r['family'] for r in kept if r['split']==s)) for s in ('train','val')},
            'source_group_overlap':0}
    (args.out/'complete.json').write_text(json.dumps(result,indent=2))
    print(result['counts'],result['family_counts'],'rejected',len(result['rejected']),flush=True)


if __name__=='__main__':main()
