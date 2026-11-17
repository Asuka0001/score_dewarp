"""Separate predicted-coordinate failures from map inversion on a fixed CPMS run.

Uses exactly that run's checkpoint hashes, image resize and frozen input traces.
No parameter is fitted or changed. This is a failure diagnostic on exposed data.
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scoredewarp.eval.geometry import network_points
from scoredewarp.eval.real_score import trace_metrics
from scoredewarp.learn.infer import load
from evaluate_cpms import sha, save


def field_folds(field, page):
    dx=field[:-1,1:]-field[:-1,:-1]
    dy=field[1:,:-1]-field[:-1,:-1]
    det=dx[...,0]*dy[...,1]-dx[...,1]*dy[...,0]
    finite=np.isfinite(det)
    cells=page[:-1,:-1]&page[:-1,1:]&page[1:,:-1]&page[1:,1:]&finite
    return {'all_finite_cells_nonpositive_fraction':float((det[finite]<=0).mean()) if finite.any() else None,
            'predicted_page_cells':int(cells.sum()),
            'predicted_page_nonpositive_fraction':float((det[cells]<=0).mean()) if cells.any() else None}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--experiment',type=Path,required=True)
    ap.add_argument('--download-root',type=Path,default=Path('data/external/cpms'))
    args=ap.parse_args();exp=args.experiment
    if not (exp/'evaluation.json').exists():ap.error('Finish evaluate_cpms.py before diagnosing fields')
    if (exp/'field_diagnostic.json').exists():ap.error('Diagnostic output exists')
    protocol=json.loads((exp/'protocol.json').read_text())
    source=Path(json.loads((args.download_root/'download_manifest.json').read_text())['source'])
    cv2.setNumThreads(1);torch.set_num_threads(2)
    records=[]
    for method,info in protocol['checkpoints'].items():
        checkpoint=ROOT/info['path']
        if sha(checkpoint)!=info['sha256']:raise ValueError('Checkpoint mismatch')
        net,size=load(checkpoint)
        for index,sample in enumerate(protocol['photos']):
            path=source/'data'/(sample['sample']+'.jpeg')
            if sha(path)!=sample['photo_sha256']:raise ValueError('Photo mismatch')
            image=cv2.imread(str(path));width=protocol['working_width']
            image=cv2.resize(image,(width,round(image.shape[0]*width/image.shape[1])),interpolation=cv2.INTER_AREA)
            photo=cv2.resize(image,tuple(size),interpolation=cv2.INTER_AREA)[...,::-1].astype(np.float32)/255
            with torch.no_grad():output=net(torch.from_numpy(photo.transpose(2,0,1).copy())[None])
            field=output['fwd'][0].cpu().numpy().transpose(1,2,0)
            page=output['page'][0,0].sigmoid().cpu().numpy()>.5
            refs=[np.asarray(v) for v in json.loads((exp/sample['sample']/'source_traces.json').read_text())]
            coords=[];keep=[]
            for staff in refs:
                p,k=network_points(output,staff,(image.shape[1],image.shape[0]),protocol['network_output_size'][1]/width)
                coords.append(p);keep.append(k)
            record={'sample':sample['sample'],'method':method,'field_folds':field_folds(field,page),
                    'direct_forward_traces':trace_metrics(coords,keep)}
            records.append(record)
            if (index+1)%15==0:print(method,index+1,'/',len(protocol['photos']),flush=True)
    summary={}
    for method in protocol['checkpoints']:
        rows=[r for r in records if r['method']==method]
        summary[method]={}
        for section,keys in [('field_folds',['all_finite_cells_nonpositive_fraction','predicted_page_nonpositive_fraction']),
                             ('direct_forward_traces',['point_coverage','staff_straight','staff_spacing_cv'])]:
            summary[method][section]={key:float(np.median([r[section][key] for r in rows if r[section][key] is not None]))
                                      for key in keys if any(r[section][key] is not None for r in rows)}
    save(exp/'field_diagnostic.json',{'summary':summary,'per_sample':records,'training_or_tuning':False,
                                     'source_sha256':sha(Path(__file__))})
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
