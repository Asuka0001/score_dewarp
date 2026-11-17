"""Long mixed-source geometry training with balanced validation and full resume.

Every epoch resamples all four source families with declared weights. Crops and
illumination are regenerated every draw. No early stopping. CPMS test is never
used for model selection; checkpoints are experimental, not deployment-ready.
"""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader,Sampler

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scoredewarp.learn.diverse_dataset import GeometryCacheDataset
from scoredewarp.learn.infer import load
from scoredewarp.learn.robust_geometry import geometry_loss
from train import plot_history

FAMILY_WEIGHTS={'deepscores':.35,'doc3d':.15,'synthetic':.30,'cpms_texture':.20}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path,data):
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False));temp.replace(path)


def save_checkpoint(path,data):
    temp=path.with_name(path.name+'.tmp');torch.save(data,temp);temp.replace(path)


class EpochSampler(Sampler):
    def __init__(self,rows,seed):
        self.rows=rows;self.seed=seed;self.epoch=0
        counts={family:sum(r['family']==family for r in rows) for family in FAMILY_WEIGHTS}
        if any(n==0 for n in counts.values()):raise ValueError('All four training families are required')
        self.weights=np.array([FAMILY_WEIGHTS[r['family']]/counts[r['family']] for r in rows])
        self.weights/=self.weights.sum()
    def __len__(self):return len(self.rows)
    def __iter__(self):
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch]))
        indices=rng.choice(len(self.rows),len(self.rows),replace=True,p=self.weights)
        seeds=rng.integers(0,2**63-1,len(indices),dtype=np.int64)
        return iter((int(i),int(s)) for i,s in zip(indices,seeds))


@torch.no_grad()
def validate(net,loader,rows,device):
    net.eval();groups=defaultdict(list)
    for batch in loader:
        indices=batch.pop('record_index').tolist()
        batch={k:v.to(device) for k,v in batch.items()}
        output=net(batch['image']);p=output['fwd'];g=batch['fwd'];page=batch['page'][:,0]>.5
        error=torch.linalg.vector_norm(p-g,dim=1)
        count=page.flatten(1).sum(1).clamp_min(1)
        epe=(error*page).flatten(1).sum(1)/count*100
        padded=torch.nn.functional.pad(page[:,None].float(),(4,4,4,4),value=0)
        eroded=-torch.nn.functional.max_pool2d(-padded,9,stride=1)[:,0]
        edge=page&(~(eroded>.5));edge_count=edge.flatten(1).sum(1).clamp_min(1)
        edge_epe=(error*edge).flatten(1).sum(1)/edge_count*100
        predicted=output['page'][:,0].sigmoid()>.5
        coverage=(predicted&page).flatten(1).sum(1)/count
        iou=(predicted&page).flatten(1).sum(1)/(predicted|page).flatten(1).sum(1).clamp_min(1)
        def determinant(field):
            dx=field[:,:,:-1,1:]-field[:,:,:-1,:-1];dy=field[:,:,1:,:-1]-field[:,:,:-1,:-1]
            return dx[:,0]*dy[:,1]-dx[:,1]*dy[:,0]
        target_det=determinant(g);det=determinant(p)
        cells=page[:,:-1,:-1]&page[:,:-1,1:]&page[:,1:,:-1]&page[:,1:,1:]&(target_det>0)
        folds=((det<=0)&cells).flatten(1).sum(1)/cells.flatten(1).sum(1).clamp_min(1)
        metrics=torch.stack([epe,edge_epe,coverage,iou,folds],1).cpu().numpy()
        for index,values in zip(indices,metrics):
            row=rows[index];key=(row['family'],row['kind'],row['profile'])
            groups[key].append(values)
    named={f'{a}/{b}/{c}':dict(zip(('coord_epe_pct','edge_epe_pct','coverage','page_iou','fold_rate'),np.mean(v,0).astype(float).tolist()))
           for (a,b,c),v in groups.items()}
    families={}
    for family in FAMILY_WEIGHTS:
        group=[v for k,v in named.items() if k.startswith(family+'/')]
        if not group:raise ValueError(f'Missing validation family {family}')
        families[family]={key:float(np.mean([v[key] for v in group])) for key in group[0]}
    balanced={key:float(np.mean([v[key] for v in families.values()])) for key in next(iter(families.values()))}
    score=balanced['coord_epe_pct']+.5*balanced['edge_epe_pct']+100*balanced['fold_rate']
    return {'balanced':balanced,'families':families,'groups':named,'selection_score':score}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache',type=Path,default=Path('data/prepared/diverse_cache_20261115'))
    ap.add_argument('--out',type=Path,default=Path('outputs/diverse_training_20261115/model'))
    ap.add_argument('--init',type=Path,default=Path('outputs/auxiliary_20261115/original_kind/validated.pt'))
    ap.add_argument('--epochs',type=int,default=120)
    ap.add_argument('--bs',type=int,default=8)
    ap.add_argument('--lr',type=float,default=2e-4)
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--seed',type=int,default=19)
    ap.add_argument('--device',choices=('mps','cuda','cpu'),default='mps')
    ap.add_argument('--fold-weight',type=float,default=.2)
    ap.add_argument('--edge-weight',type=float,default=.5)
    ap.add_argument('--consistency-weight',type=float,default=5.)
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    if min(args.epochs,args.bs)<1 or args.workers<0:ap.error('Invalid training counts')
    if args.device=='mps' and not torch.backends.mps.is_available():ap.error('MPS is unavailable; run in a hardware-enabled environment')
    args.out.mkdir(parents=True,exist_ok=True)
    lock=(args.out/'training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    last=args.out/'last.pt'
    if last.exists() and not args.resume:ap.error('Training exists; use --resume or a fresh output')
    if args.resume and not last.exists():ap.error('No last checkpoint to resume')
    info=json.loads((args.cache/'complete.json').read_text())
    train_rows=[r for r in info['records'] if r['split']=='train'];val_rows=[r for r in info['records'] if r['split']=='val']
    config={**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='resume'},
            'cache_manifest_sha256':sha(args.cache/'complete.json'),'initial_checkpoint_sha256':sha(args.init),
            'size':info['size'],'train_samples':len(train_rows),'val_samples':len(val_rows),'family_weights':FAMILY_WEIGHTS,
            'early_stopping':False,'selection':'equal-family mean (EPE% + 0.5 edge EPE% + 100 fold fraction); equal kind/profile within family',
            'cpms_test_used_for_training_or_selection':False,'deployment_ready':False,
            'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'scoredewarp/learn/robust_geometry.py',
             ROOT/'scoredewarp/learn/diverse_dataset.py',ROOT/'scoredewarp/learn/model.py']}}
    if args.resume:
        if json.loads((args.out/'config.json').read_text())!=config:ap.error('Resume config/source/data mismatch')
    else:
        save_json(args.out/'config.json',config)
        snapshot=args.out/'source';snapshot.mkdir(exist_ok=True)
        for path in config['source_sha256']:(snapshot/path.replace('/','__')).write_bytes((ROOT/path).read_bytes())
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(4);cv2.setNumThreads(1)
    net,size=load(args.init,args.device)
    for head in (net.head_kind,net.head_seg,net.head_struct):head.requires_grad_(False)
    # Backbone changes invalidate all previous auxiliary deployment admissions.
    auxiliary={k:{'trained':False,'ready':False,'reason':'geometry backbone retrained; revalidation required'} for k in ('kind','seg','struct')}
    optimizer=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=args.lr,weight_decay=1e-4)
    sampler=EpochSampler(train_rows,args.seed)
    common={'batch_size':args.bs,'num_workers':args.workers}
    if args.workers:common.update(multiprocessing_context='spawn',persistent_workers=True,prefetch_factor=2)
    train_loader=DataLoader(GeometryCacheDataset(train_rows,True),sampler=sampler,**common)
    val_loader=DataLoader(GeometryCacheDataset(val_rows),**common)
    total_steps=args.epochs*len(train_loader);warmup=2*len(train_loader)
    def schedule(step):
        if step<warmup:return .05+.95*step/max(warmup,1)
        fraction=(step-warmup)/max(total_steps-warmup,1)
        return .01+.99*.5*(1+math.cos(math.pi*min(fraction,1)))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    history=[];best=float('inf');step=0;start_epoch=1;visits=np.zeros(len(train_rows),np.int64)
    if args.resume:
        state=torch.load(last,map_location=args.device,weights_only=True)
        net.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        step=state['step'];start_epoch=state['epoch']+1;best=state['best_score']
        history=json.loads((args.out/'history.json').read_text());visits=np.asarray(state['visits'],np.int64)
        torch.set_rng_state(state['rng_state'].cpu())
    started=time.time();elapsed_previous=history[-1].get('elapsed_s',0) if history else 0
    def status(**changes):
        save_json(args.out/'status.json',{'status':'running','pid':os.getpid(),'device':args.device,
                  'planned_epochs':args.epochs,'step':step,'elapsed_s':elapsed_previous+time.time()-started,
                  'updated_at':datetime.now(timezone.utc).isoformat(),**changes})
    status(phase='initial_validation',epoch=start_epoch-1)
    if not args.resume:
        initial=validate(net,val_loader,val_rows,args.device)
        history.append({'epoch':0,'step':0,'elapsed_s':time.time()-started,'validation':initial,
                        'val_coord_epe_pct':initial['balanced']['coord_epe_pct'],'val_page_iou':initial['balanced']['page_iou']})
        save_json(args.out/'history.json',history)
        print('initial validation',json.dumps(initial['families']),flush=True)
    for epoch in range(start_epoch,args.epochs+1):
        net.train();sampler.epoch=epoch;count=0;parts_sum=defaultdict(float);loss_sum=0.
        epoch_start=time.time();status(phase='training',epoch=epoch)
        for batch_index,batch in enumerate(train_loader):
            indices=batch.pop('record_index').numpy();np.add.at(visits,indices,1)
            batch={k:v.to(args.device) for k,v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output=net(batch['image'])
            loss,parts=geometry_loss(output,batch,args.fold_weight,args.edge_weight,args.consistency_weight)
            if not torch.isfinite(loss):raise ValueError(f'Nonfinite loss at step {step}')
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);optimizer.step();scheduler.step()
            n=len(indices);count+=n;loss_sum+=loss.item()*n
            for key,value in parts.items():parts_sum[key]+=value.detach().item()*n
            step+=1
            if step%25==0 or batch_index==0:
                status(phase='training',epoch=epoch,batch=batch_index+1,batches=len(train_loader),
                       loss=loss.item(),distinct_training_samples_seen=int((visits>0).sum()))
                print('epoch',epoch,'batch',batch_index+1,'/',len(train_loader),'step',step,
                      'loss',round(loss.item(),6),'elapsed_s',round(time.time()-started),flush=True)
        row={'epoch':epoch,'step':step,'train_loss':loss_sum/count,'train_parts':{k:v/count for k,v in parts_sum.items()},
             'lr':optimizer.param_groups[0]['lr'],'elapsed_s':elapsed_previous+time.time()-started,
             'epoch_training_s':time.time()-epoch_start,'distinct_training_samples_seen':int((visits>0).sum())}
        improved=False
        if epoch==1 or epoch%2==0 or epoch==args.epochs:
            status(phase='validation',epoch=epoch)
            metrics=validate(net,val_loader,val_rows,args.device);row['validation']=metrics
            row.update(val_coord_epe_pct=metrics['balanced']['coord_epe_pct'],val_page_iou=metrics['balanced']['page_iou'])
            if metrics['selection_score']<best:best=metrics['selection_score'];improved=True
        history.append(row)
        state={'model':net.state_dict(),'arch':net.arch,'size':size,'optimizer':optimizer.state_dict(),
               'scheduler':scheduler.state_dict(),'epoch':epoch,'step':step,'best_score':best,
               'config':config,'auxiliary_heads':auxiliary,'rng_state':torch.get_rng_state(),'visits':visits.tolist()}
        save_checkpoint(last,state)
        if improved:save_checkpoint(args.out/'best.pt',state)
        if epoch%20==0:save_checkpoint(args.out/f'epoch_{epoch:03d}.pt',state)
        save_json(args.out/'history.json',history);plot_history(history,args.out/'curves.png')
        status(phase='epoch_complete',epoch=epoch,best_score=best,distinct_training_samples_seen=int((visits>0).sum()))
        print('epoch complete',json.dumps(row),flush=True)
    save_json(args.out/'report.json',{'status':'complete','epochs_completed':args.epochs,'steps':step,
              'best_score':best,'distinct_training_samples_seen':int((visits>0).sum()),
              'train_samples':len(train_rows),'early_stopping':False,'deployment_ready':False})
    status(status='complete',phase='complete',epoch=args.epochs,best_score=best)


if __name__=='__main__':main()
