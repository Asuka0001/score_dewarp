"""Cached mixed-source geometry data with fresh, coordinate-consistent crops."""
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def augment(sample,seed):
    """Crop input and dense targets together, then normalize visible flat extent.

    The target remains a planar coordinate system under an affine change of
    origin/scale. Never infer geometry from real-photo pixels without a target.
    """
    rng=np.random.default_rng(seed)
    result={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in sample.items()}
    _,h,w=sample['image'].shape
    if rng.random()<.65 and (sample['page'][0]>.5).any():
        ys,xs=np.nonzero(sample['page'][0]>.5)
        if rng.random()<.7:
            x0,y0,x1,y1=xs.min(),ys.min(),xs.max()+1,ys.max()+1
            padx=int((x1-x0)*rng.uniform(-.08,.08));pady=int((y1-y0)*rng.uniform(-.08,.08))
            x0=max(0,x0-padx);x1=min(w,x1+padx);y0=max(0,y0-pady);y1=min(h,y1+pady)
        else:
            x0=int(w*rng.uniform(0,.15));x1=int(w*rng.uniform(.85,1))
            y0=int(h*rng.uniform(0,.15));y1=int(h*rng.uniform(.85,1))
        if x1-x0>=w*.4 and y1-y0>=h*.4:
            area=sample['page'][0,y0:y1,x0:x1]
            denominator=cv2.resize(area,(w,h),interpolation=cv2.INTER_LINEAR)
            page=(denominator>.999).astype(np.float32)
            field=cv2.resize((sample['fwd'][:,y0:y1,x0:x1]*area).transpose(1,2,0),(w,h),interpolation=cv2.INTER_LINEAR)
            field=field/np.maximum(denominator[:,:,None],1e-6)
            if page.mean()>.05:
                lo=field[page>0].min(0);hi=field[page>0].max(0)
                if np.min(hi-lo)>.1:
                    field=(field-lo)/(hi-lo)
                    field[page==0]=0
                    result['fwd']=np.ascontiguousarray(field.transpose(2,0,1),dtype=np.float32)
                    result['page']=page[None]
                    result['image']=cv2.resize(sample['image'][:,y0:y1,x0:x1].transpose(1,2,0),(w,h),interpolation=cv2.INTER_AREA).transpose(2,0,1)
                    for key in ('inst_h','inst_v'):
                        result[key]=cv2.resize(sample[key][y0:y1,x0:x1].astype(np.float32),(w,h),interpolation=cv2.INTER_NEAREST).astype(np.int64)
                        result[key]*=page.astype(np.int64)
    image=result['image'].transpose(1,2,0)
    yy,xx=np.mgrid[:h,:w].astype(np.float32)
    light=1+rng.uniform(-.3,.3)*(xx/w-.5)+rng.uniform(-.3,.3)*(yy/h-.5)
    image=np.clip(image*light[:,:,None]*rng.uniform(.85,1.1),0,1)
    image=image**rng.uniform(.85,1.15)
    image+=rng.normal(0,rng.uniform(0,.012),image.shape).astype(np.float32)
    result['image']=np.ascontiguousarray(np.clip(image,0,1).transpose(2,0,1),dtype=np.float32)
    return result


class GeometryCacheDataset(Dataset):
    def __init__(self,rows,training=False):self.rows=rows;self.training=training
    def __len__(self):return len(self.rows)
    def __getitem__(self,request):
        if isinstance(request,tuple):index,seed=request
        else:index,seed=request,None
        with np.load(self.rows[index]['cache']) as z:
            sample={'image':z['image'].astype(np.float32)/255,'fwd':z['fwd'].astype(np.float32),
                    'page':z['page'].astype(np.float32),'inst_h':z['inst_h'].astype(np.int64),
                    'inst_v':z['inst_v'].astype(np.int64)}
        if self.training and seed is not None:sample=augment(sample,seed)
        return {**{k:torch.from_numpy(np.ascontiguousarray(v)) for k,v in sample.items()},
                'record_index':index}
