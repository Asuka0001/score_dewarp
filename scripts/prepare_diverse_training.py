"""Build diverse paired geometry data from local training sources only.

Uses full DeepScores source pages, fresh three-kind rendering and teacher-
rectified CPMS official training strips. CPMS test photos never become training
textures. Existing data are not modified. All new physical warps are screened.
"""
import argparse
from collections import defaultdict
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
from scoredewarp.core.pipeline import _detect
from scoredewarp.core.preprocess import normalize_illumination
from scoredewarp.synth.distort import simulate_photo
from scoredewarp.synth.render import render_page

MODES=('identity','mild','tight','standard','shadow','blur')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strip_texture(job):
    path,destination=map(Path,job);cv2.setNumThreads(1)
    image=cv2.imread(str(path),0)
    h,w=image.shape;image=cv2.resize(image,(1100,round(h*1100/w)),interpolation=cv2.INTER_AREA)
    _,_,structures=_detect(image)
    candidates=[s for s in structures.staves if len(s.tuples)>=20]
    if not candidates:return {'source':str(path),'accepted':False,'reason':'no long 5-line system'}
    staff=max(candidates,key=lambda s:len(s.tuples));p=np.asarray(staff.tuples)
    x=p[:,:,0].mean(1);center=p[:,:,1].mean(1);spacing=np.diff(p[:,:,1],axis=1).mean(1)
    order=np.argsort(x);x,center,spacing=x[order],center[order],spacing[order]
    if x[-1]-x[0]<.6*image.shape[1] or np.min(spacing)<=0:
        return {'source':str(path),'accepted':False,'reason':'short/invalid reference'}
    cy=float(np.median(center));step=float(np.median(spacing))
    yy,xx=np.mgrid[:image.shape[0],:image.shape[1]].astype(np.float32)
    curve=np.interp(np.arange(image.shape[1]),x,center)
    gaps=np.interp(np.arange(image.shape[1]),x,spacing)
    source_y=(yy-cy)*gaps[None]/step+curve[None]
    paper=int(np.percentile(image,85))
    rectified=cv2.remap(image,xx,source_y.astype(np.float32),cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT,borderValue=paper)
    _,_,check=_detect(rectified)
    good=[]
    for s in check.staves:
        if len(s.tuples)<20:continue
        q=np.asarray(s.tuples);scale=float(np.median(np.diff(q[:,:,1],axis=1)))
        if scale>0:good.append(float(np.median(q[:,:,1].std(0))/scale))
    if not good or min(good)>.2:
        return {'source':str(path),'accepted':False,'reason':'teacher rectification quality gate'}
    destination.parent.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(destination),rectified)
    return {'source':str(path),'source_sha256':digest(path),'accepted':True,'texture':str(destination),
            'texture_sha256':digest(destination),'cy':cy,'spacing':step,'x_range':[float(x[0]),float(x[-1])],
            'teacher_residual':min(good)}


def identity_photo(flat,ann,rng,size):
    w,h=size;hf,wf=flat.shape
    image=cv2.resize(flat,(w,h),interpolation=cv2.INTER_AREA).astype(np.float32)
    yy,xx=np.mgrid[:h,:w].astype(np.float32)
    light=1+rng.uniform(-.18,.18)*(xx/w-.5)+rng.uniform(-.18,.18)*(yy/h-.5)
    image=np.clip(image*light+rng.normal(0,rng.uniform(0,2),image.shape),0,255).astype(np.uint8)
    image=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    mapping=np.stack([xx[::2,::2]*wf/w,yy[::2,::2]*hf/h],-1)
    annotations=[{**a,'pts':(np.asarray(a['pts'])*[w/wf,h/hf]).tolist()} for a in ann]
    return {'image':image,'ann':annotations,'bmap_lo':mapping,'projection':{'valid':True,'type':'identity'}}


def mosaic(rows,rng):
    # Remove per-strip lighting before composing; otherwise patch boundaries
    # create artificial horizontal shadow bands absent from the geometry GT.
    paper=255
    flat=np.full((1754,1240),paper,np.uint8);ann=[]
    count=int(rng.integers(6,11));choice=rng.choice(len(rows),count,replace=len(rows)<count)
    centers=np.linspace(160,1594,count)
    for staff_id,(i,ycenter) in enumerate(zip(choice,centers)):
        r=rows[i];strip=normalize_illumination(cv2.imread(r['texture'],0))
        top=int(round(ycenter-r['cy']));left=70
        y0=max(top,0);y1=min(top+strip.shape[0],flat.shape[0])
        if y1<=y0:continue
        flat[y0:y1,left:left+strip.shape[1]]=strip[y0-top:y1-top]
        xs=np.linspace(left+r['x_range'][0],left+r['x_range'][1],100)
        for line in range(5):
            y=top+r['cy']+(line-2)*r['spacing']
            ann.append({'kind':'staff_line','pts':np.stack([xs,np.full_like(xs,y)],1).tolist(),
                        'staff':staff_id,'line':line,'spacing':r['spacing']})
    return flat,ann,{'kind':'staff','size':[1240,1754]}


def generate(job):
    cv2.setNumThreads(1)
    seed=job['seed'];rng=np.random.default_rng(seed)
    if job['type']=='render':flat,ann,meta=render_page(job['kind'],rng)
    elif job['type']=='deepscores':
        flat=cv2.imread(job['flat'],0);ann=[];meta={'kind':'staff','size':[flat.shape[1],flat.shape[0]]}
    else:flat,ann,meta=mosaic(job['textures'],rng)
    mode=job['mode'];size=(768,1024)
    if mode=='identity':sim=identity_photo(flat,ann,rng,size)
    else:
        settings={
            'mild':dict(strength=float(rng.uniform(.05,.25)),fill_range=(.96,1.08)),
            'tight':dict(strength=float(rng.uniform(.15,.7)),fill_range=(1.08,1.4)),
            'standard':dict(strength=float(rng.uniform(.6,1.3)),fill_range=(.72,.98)),
            'shadow':dict(strength=float(rng.uniform(.2,1.)),fill_range=(.92,1.2),shadow_probability=1.),
            'blur':dict(strength=float(rng.uniform(.2,1.)),fill_range=(.92,1.2),blur_sigma=1.8)}
        sim=simulate_photo(flat,ann,rng,out_wh=size,require_invertible=True,**settings[mode])
    dest=Path(job['destination']);dest.mkdir(parents=True)
    cv2.imwrite(str(dest/'photo.jpg'),sim['image'],[cv2.IMWRITE_JPEG_QUALITY,95])
    cv2.imwrite(str(dest/'flat.png'),flat)
    # Float32 avoids introducing extra zero Jacobians through float16 rounding.
    np.save(dest/'gt_bmap_s2.npy',sim['bmap_lo'].astype(np.float32))
    full_supervision=job['type']=='render'
    meta.update(source=job['source'],source_group=job['group'],split=job['split'],seed=seed,profile=mode,
                projection=sim['projection'],supervision={'kind':True,'seg':full_supervision,'struct':full_supervision},
                texture_provenance='teacher-rectified CPMS training mosaics; not native dense GT' if job['type']=='cpms' else job['type'])
    (dest/'gt.json').write_text(json.dumps({'meta':meta,'flat_ann':ann,'photo_ann':sim['ann']}))
    return {'path':str(dest),'split':job['split'],'source':job['source'],'source_group':job['group'],
            'profile':mode,'kind':meta['kind'],'projection':sim['projection'],
            'sha256':{p.name:digest(p) for p in dest.iterdir()}}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=Path('data/prepared/diverse_20261115'))
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--seed',type=int,default=2026111519)
    ap.add_argument('--render-train-per-cell',type=int,default=80)
    args=ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()):ap.error('Use a fresh empty output directory')
    args.out.mkdir(parents=True)
    cpms=Path(json.loads((ROOT/'data/external/cpms/download_manifest.json').read_text())['source'])
    train_paths=sorted((cpms/'semantic/training').glob('*/*.jpg'))
    test_ids={p.stem for p in (cpms/'data').glob('*.jpeg')}
    if {p.parent.name.split('-')[0] for p in train_paths}&test_ids:raise ValueError('CPMS source ID overlap')
    # Join identical note sequences across source IDs before splitting textures.
    parent={p.parent.name.split('-')[0]:p.parent.name.split('-')[0] for p in train_paths}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    def notes(p):
        tokens=p.read_text().split()
        return tuple(t for t in tokens if t.startswith('note-'))
    test_notes={notes(p) for p in (cpms/'semantic/test').glob('*/*.semantic')}
    owners={};usable=[];excluded=[]
    for p in train_paths:
        semantic=p.with_suffix('.semantic')
        if not semantic.exists():excluded.append({'path':str(p),'reason':'missing semantic content audit'});continue
        sequence=notes(semantic)
        if sequence in test_notes:excluded.append({'path':str(p),'reason':'note sequence overlaps CPMS test'});continue
        group=p.parent.name.split('-')[0]
        if sequence in owners:parent[find(group)]=find(owners[sequence])
        owners[sequence]=group;usable.append(p)
    assignments={key:('val' if int(hashlib.sha256((str(args.seed)+':'+find(key)).encode()).hexdigest()[:8],16)%5==0 else 'train') for key in parent}
    texture_jobs=[(str(p),str(args.out/'textures'/(p.stem+'.png'))) for p in usable]
    protocol={'seed':args.seed,'profiles':list(MODES),'render_train_per_cell':args.render_train_per_cell,
              'deepscores_splits':'preserve full_v3 source-group assignment',
              'cpms_split_rule':'source-ID components joined by exact note sequence; hash 80/20',
              'cpms_source_assignments':assignments,'cpms_excluded':excluded,'workers':args.workers,
              'cpms_test_training':False,'old_datasets_modified':False,
              'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in
                  [Path(__file__),ROOT/'scoredewarp/synth/distort.py',ROOT/'scoredewarp/synth/render.py',ROOT/'scoredewarp/core/structures.py']}}
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    textures=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for result in pool.map(strip_texture,texture_jobs):
            textures.append(result)
            if len(textures)%100==0:print('teacher textures',len(textures),'/',len(texture_jobs),flush=True)
    (args.out/'texture_audit.json').write_text(json.dumps(textures,indent=2))
    by_group=defaultdict(list)
    for r in textures:
        if r['accepted']:by_group[Path(r['source']).parent.name.split('-')[0]].append(r)
    jobs=[]
    def add(split,kind,mode,group,source,typ,**extra):
        i=len(jobs);seed=int(np.random.SeedSequence([args.seed,i]).generate_state(1,dtype=np.uint64)[0])
        jobs.append({'seed':seed,'destination':str(args.out/split/f'{source}_{i:06d}'),
                     'split':split,'kind':kind,'mode':mode,'group':group,'source':source,'type':typ,**extra})
    ds=ROOT/'data/prepared/deepscores_full_v3';manifest=json.loads((ds/'manifest.json').read_text())
    seen=set()
    for r in manifest['samples']:
        split=r['path'].split('/')[0]
        if split=='test' or (split,r['source_file']) in seen:continue
        seen.add((split,r['source_file']))
        for mode in (('identity','mild','tight') if split=='train' else ('identity','tight')):
            add(split,'staff',mode,r['source_group'],'deepscores','deepscores',flat=str(ds/r['path']/'flat.png'))
    for group,rows in sorted(by_group.items()):
        split=assignments[group]
        for index in range(6 if split=='train' else 2):
            for mode in ('identity','tight'):
                add(split,'staff',mode,find(group),'cpms_training_texture','cpms',textures=rows)
    for split,count in (('train',args.render_train_per_cell),('val',12)):
        for kind in ('staff','jianpu','mixed'):
            for mode in MODES:
                for index in range(count):
                    add(split,kind,mode,f'diverse-{split}-{kind}-{mode}-{index}-{args.seed}','diverse_synth','render')
    (args.out/'generation_jobs.json').write_text(json.dumps(jobs,indent=2))
    print('generated job counts',dict((s,sum(j['split']==s for j in jobs)) for s in ('train','val')),flush=True)
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for row in pool.map(generate,jobs):
            rows.append(row)
            if len(rows)%100==0:print('paired samples',len(rows),'/',len(jobs),flush=True)
    train_groups={(r['source'],r['source_group']) for r in rows if r['split']=='train'}
    val_groups={(r['source'],r['source_group']) for r in rows if r['split']=='val'}
    if train_groups&val_groups:raise ValueError('Generated source-group leakage')
    result={'counts':{s:sum(r['split']==s for r in rows) for s in ('train','val')},'samples':rows,
            'source_group_overlap':0,'cpms_test_used_for_training':False}
    (args.out/'manifest.json').write_text(json.dumps(result,indent=2))
    print(result['counts'],flush=True)


if __name__=='__main__':main()
