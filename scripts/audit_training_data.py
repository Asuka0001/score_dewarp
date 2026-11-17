"""Inventory local datasets and audit sampled geometry without changing data."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


def map_quality(folder):
    gt=json.loads((folder/'gt.json').read_text())
    field=np.load(folder/'gt_bmap_s2.npy').astype(np.float64)
    # Ignore off-page coordinates and NaNs when checking sampled orientation.
    bounds=np.asarray(gt['meta']['size'])
    valid=np.isfinite(field).all(-1)&(field>=0).all(-1)&(field<bounds).all(-1)
    dx=field[:-1,1:]-field[:-1,:-1];dy=field[1:,:-1]-field[:-1,:-1]
    det=dx[...,0]*dy[...,1]-dx[...,1]*dy[...,0]
    cells=valid[:-1,:-1]&valid[:-1,1:]&valid[1:,:-1]&valid[1:,1:]
    return {'page_fraction':float(valid.mean()),'cells':int(cells.sum()),
            'nonpositive_fraction':float((det[cells]<=0).mean()) if cells.any() else None}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path('data'))
    ap.add_argument('--out',type=Path,default=Path('outputs/diverse_training_20261115/audit.json'))
    args=ap.parse_args();args.out.parent.mkdir(parents=True,exist_ok=True)
    inventory={}
    for manifest in sorted((args.root/'prepared').glob('*/manifest.json')):
        data=json.loads(manifest.read_text());record={'manifest':str(manifest),'counts':data['counts'],'splits':{}}
        for split in ('train','val','test'):
            rows=[r for r in data['samples'] if r['path'].startswith(split+'/')]
            record['splits'][split]={'samples':len(rows),'groups':len({r['source_group'] for r in rows}),
                                     'source_files':len({r.get('source_file',r['source_group']) for r in rows})}
        inventory[manifest.parent.name]=record
    selected={}
    for dataset in ('deepscores_full_v3','doc3d_pilot','structure_suite_20261115_v2'):
        root=args.root/'prepared'/dataset
        data=json.loads((root/'manifest.json').read_text());records=[]
        for row in data['samples']:
            if not row['path'].startswith('train/'):continue
            quality=map_quality(root/row['path'])
            records.append({'path':str(root/row['path']),'source_group':row['source_group'],**quality})
            if len(records)%500==0:print(dataset,len(records),flush=True)
        selected[dataset]={'samples':records,'summary':{
            'samples':len(records),'no_valid_cells':sum(r['cells']==0 for r in records),
            'nonpositive_gt_pages':sum((r['nonpositive_fraction'] or 0)>0 for r in records),
            'gt_fold_above_0_1pct':sum((r['nonpositive_fraction'] or 0)>.001 for r in records),
            'page_fraction_median':float(np.median([r['page_fraction'] for r in records]))}}
        print(dataset,selected[dataset]['summary'],flush=True)
    source=Path(json.loads((args.root/'external/cpms/download_manifest.json').read_text())['source'])
    cpms={s:{'images':len(list((source/'semantic'/s).rglob('*.jpg')))+len(list((source/'semantic'/s).rglob('*.jpeg'))),
             'source_ids':len({p.parent.name.split('-')[0] for p in (source/'semantic'/s).rglob('*') if p.suffix in ('.jpg','.jpeg')})}
          for s in ('training','test')}
    result={'inventory':inventory,'training_geometry_audit':selected,'cpms':cpms,
            'excluded':'pilot duplicate pages, unsafe v1 suite, and all prior val/test/debug benchmarks are not extra training data'}
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(args.out,flush=True)


if __name__=='__main__':main()
