"""Fit spatial seg/struct branches while preserving original geometry/kind.

This development follow-up addresses the measured failure of the linear
pixel probes. RGB + frozen decoder features feed independent spatial heads.
"""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scoredewarp.learn.infer import load
from scoredewarp.learn.model import ScoreDewarpNet
from scoredewarp.learn.dataset import build_targets
from scoredewarp.eval.auxiliary import segmentation_metrics,heatmap_metrics
from train_auxiliary import TRAIN,VAL_ROOTS,sha,save_json,geometry_fingerprint


@lru_cache(maxsize=400)
def targets(path):
    with np.load(path) as data:
        return {key:data[key] for key in ('seg','struct')}


def prepare(exp,device):
    cache=exp/'spatial_cache'
    if (cache/'complete.json').exists():return
    if cache.exists() and any(cache.iterdir()):raise ValueError('Use a fresh spatial cache')
    cache.mkdir()
    net,size=load(exp/'original_kind/best.pt',device)
    rows=[]
    paths=[('train',p) for p in sorted((ROOT/TRAIN).iterdir()) if (p/'gt.json').exists()]
    paths += [('val',p) for p in sorted((ROOT/VAL_ROOTS['synthetic']).iterdir()) if (p/'gt.json').exists()]
    with torch.no_grad():
        for split,path in paths:
            d=build_targets(str(path),tuple(size))
            if not(d['seg_valid'] and d['struct_valid']):continue
            dest=cache/(split+'_'+path.name)
            image=torch.from_numpy(d['image'])[None].to(device)
            if split=='val':
                dense=np.load(exp/'cache_original/dense_val'/(path.name+'.npy')).astype(np.float32)
            else:
                dense=net.forward_features(image)[0][0].cpu().numpy()
            np.save(str(dest)+'.npy',np.concatenate([dense,d['image']]).astype(np.float16))
            np.savez_compressed(str(dest)+'.targets.npz',seg=d['seg'].astype(np.uint8),struct=d['struct'].astype(np.uint8))
            rows.append({'split':split,'sample':path.name,'path':dest.name})
            if len(rows)%50==0:print('spatial cache',len(rows),flush=True)
    save_json(cache/'complete.json',{'samples':rows,'size':list(size),'geometry_fingerprint':geometry_fingerprint(net)})


@torch.no_grad()
def validate(net,cache,rows,device):
    confusion=np.zeros((5,5),np.int64)
    tp,pp,tt=np.zeros(3,np.int64),np.zeros(3,np.int64),np.zeros(3,np.int64)
    for row in rows:
        x=torch.from_numpy(np.load(cache/(row['path']+'.npy')).astype(np.float32))[None].to(device)
        target=targets(str(cache/(row['path']+'.targets.npz')))
        seg=net.head_seg(x)[0].argmax(0).cpu().numpy()
        p=net.head_struct(x)[0].sigmoid().cpu().numpy()>=.5
        truth=target['struct']>0
        confusion += np.bincount((target['seg'].astype(np.int64)*5+seg).ravel(),minlength=25).reshape(5,5)
        tp += (p&truth).sum((1,2));pp+=p.sum((1,2));tt+=truth.sum((1,2))
    return {'seg':segmentation_metrics(confusion),'struct':heatmap_metrics(tp,pp,tt)}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--experiment',type=Path,default=Path('outputs/auxiliary_20261115'))
    ap.add_argument('--device',default='mps',choices=('mps','cuda','cpu'))
    args=ap.parse_args();exp=args.experiment.resolve();out=exp/'original_spatial'
    if (out/'report.json').exists():return
    if out.exists() and any(out.iterdir()):raise ValueError('Incomplete training; use fresh output')
    out.mkdir()
    init=exp/'original_kind/best.pt'
    protocol={'init_sha256':sha(init),'epochs':20,'batch':4,'lr':.001,'seed':7,
              'inputs':'RGB plus frozen 32-channel decoder features','hidden':16,
              'seg_dilations':[1,2],'struct_dilations':[1,2,4],
              'loss':'sqrt-class-balanced seg CE + 0.5 channel-balanced BCE + 0.5 soft Dice',
              'struct_pos_weight':'training-sample negative/positive ratio capped at 200',
              'validation':'full pixels, probability threshold 0.5, epochs 1 and every 2',
              'selection':'0.5 segmentation mIoU + 0.5 structure macro F1',
              'geometry_and_kind_frozen':True,'test_access':False,
              'source_sha256':{p:sha(ROOT/p) for p in ('scripts/train_spatial_auxiliary.py','scoredewarp/learn/model.py')}}
    save_json(out/'protocol.json',protocol)
    for path in protocol['source_sha256']:
        (out/path.replace('/','__')).write_bytes((ROOT/path).read_bytes())
    prepare(exp,args.device)
    torch.manual_seed(7);rng=np.random.default_rng(7)
    parent,size=load(init,args.device)
    net=ScoreDewarpNet(**{**parent.arch,'auxiliary_context':True}).to(args.device)
    state=net.state_dict()
    for key,value in parent.state_dict().items():
        if not key.startswith(('head_seg.','head_struct.')):state[key]=value
    net.load_state_dict(state)
    net.requires_grad_(False).eval()
    net.head_seg.requires_grad_(True);net.head_struct.requires_grad_(True)
    assert geometry_fingerprint(net)==geometry_fingerprint(parent)
    cache=exp/'spatial_cache';info=json.loads((cache/'complete.json').read_text())
    train=[r for r in info['samples'] if r['split']=='train'];val=[r for r in info['samples'] if r['split']=='val']
    counts=np.bincount(np.load(exp/'cache_original/train_seg.npy'),minlength=5)
    weights=torch.from_numpy(1/np.sqrt(np.maximum(counts,1))).float().to(args.device)
    weights/=weights.mean()
    labels=np.load(exp/'cache_original/train_struct.npy');positive=labels.sum(0)
    pos_weight=torch.tensor(np.minimum((len(labels)-positive)/np.maximum(positive,1),200),dtype=torch.float32,device=args.device).view(1,3,1,1)
    optimizer=torch.optim.AdamW([*net.head_seg.parameters(),*net.head_struct.parameters()],lr=.001,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=20)
    history=[];best=-1.;start=time.time()
    for ep in range(1,21):
        total=0.
        order=rng.permutation(len(train))
        for start_index in range(0,len(order),4):
            batch=[train[i] for i in order[start_index:start_index+4]]
            x=torch.from_numpy(np.stack([np.load(cache/(r['path']+'.npy')) for r in batch])).to(args.device,dtype=torch.float32)
            target=[targets(str(cache/(r['path']+'.targets.npz'))) for r in batch]
            seg=torch.from_numpy(np.stack([r['seg'] for r in target])).to(args.device,dtype=torch.long)
            struct=torch.from_numpy(np.stack([r['struct'] for r in target])).to(args.device,dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            logits=net.head_struct(x);prob=logits.sigmoid()
            dice=1-((2*(prob*struct).sum((0,2,3))+1)/(prob.sum((0,2,3))+struct.sum((0,2,3))+1)).mean()
            loss=F.cross_entropy(net.head_seg(x),seg,weight=weights)+.5*F.binary_cross_entropy_with_logits(logits,struct,pos_weight=pos_weight)+.5*dice
            if not torch.isfinite(loss):raise ValueError('Nonfinite spatial loss')
            loss.backward();optimizer.step();total+=loss.item()*len(batch)
        scheduler.step()
        row={'epoch':ep,'loss':total/len(train),'elapsed_s':time.time()-start}
        if ep==1 or ep%2==0:
            metrics=validate(net,cache,val,args.device)
            score=.5*metrics['seg']['miou']+.5*metrics['struct']['macro_f1']
            row.update(metrics=metrics,score=score)
            if score>best:
                best=score
                torch.save({'model':net.state_dict(),'arch':net.arch,'size':size,'epoch':ep,
                            'config':protocol,'metrics':metrics,'auxiliary_heads':{'kind':{'trained':True,'ready':False}}},out/'best.pt')
            print('spatial',ep,'/20','mIoU',round(metrics['seg']['miou'],4),'F1',round(metrics['struct']['macro_f1'],4),'seconds',round(row['elapsed_s']),flush=True)
        history.append(row);save_json(out/'history.json',history)
    selected,_=load(out/'best.pt',args.device)
    assert geometry_fingerprint(selected)==geometry_fingerprint(parent)
    assert all(torch.equal(v,selected.state_dict()[k]) for k,v in parent.state_dict().items() if k.startswith('head_kind.'))
    save_json(out/'report.json',{'best_score':best,'best_epoch':max((r for r in history if 'score' in r),key=lambda r:r['score'])['epoch'],
                               'geometry_and_kind_unchanged':True,'checkpoint_sha256':sha(out/'best.pt'),'elapsed_s':time.time()-start})


if __name__=='__main__':main()
