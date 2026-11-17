"""Geometry supervision with multiscale orientation and page-edge constraints."""
import torch
import torch.nn.functional as F

from .model import _instance_variance


def geometry_loss(out, batch, fold_weight=.2, edge_weight=.5, consistency_weight=5.):
    pf,gf,page=out['fwd'],batch['fwd'],batch['page']
    zero=pf.sum()*0
    parts={'fwd':((pf-gf).abs()*page).sum()/page.sum().clamp_min(1)/2,
           'page':F.binary_cross_entropy_with_logits(out['page'],page)}
    mx=page[:,:,:,1:]*page[:,:,:,:-1];my=page[:,:,1:,:]*page[:,:,:-1,:]
    parts['grad']=(((pf[:,:,:,1:]-pf[:,:,:,:-1])-(gf[:,:,:,1:]-gf[:,:,:,:-1])).abs()*mx).sum()/mx.sum().clamp_min(1)*10
    parts['grad']+=(((pf[:,:,1:,:]-pf[:,:,:-1,:])-(gf[:,:,1:,:]-gf[:,:,:-1,:])).abs()*my).sum()/my.sum().clamp_min(1)*10
    # Include photo edges even when the page fills the entire image.
    eroded=-F.max_pool2d(-F.pad(page,(4,4,4,4),value=0),9,stride=1)
    band=(page-eroded).clamp_min(0)
    parts['edge']=((pf-gf).abs()*band).sum()/band.sum().clamp_min(1)/2
    fold=zero;rates=[]
    h,w=pf.shape[-2:]
    for stride in (1,4):
        if min(h,w)<=stride:continue
        def determinant(field):
            dx=(field[:,:,:-stride,stride:]-field[:,:,:-stride,:-stride])*(w/stride)
            dy=(field[:,:,stride:,:-stride]-field[:,:,:-stride,:-stride])*(h/stride)
            return dx[:,0]*dy[:,1]-dx[:,1]*dy[:,0]
        truth=determinant(gf);pred=determinant(pf)
        valid=page[:,0,:-stride,:-stride]*page[:,0,stride:,:-stride]*page[:,0,:-stride,stride:]*page[:,0,stride:,stride:]
        # Do not force orientation using a folded/quantized target cell.
        valid=valid*(truth>1e-4)
        penalty=F.relu(.05*truth-pred)/truth.detach().clamp_min(.1)
        fold+=(penalty*valid).sum()/valid.sum().clamp_min(1)
        rates.append(((pred<=0)*valid).sum()/valid.sum().clamp_min(1))
    parts['fold']=fold/max(len(rates),1)
    parts['cons']=zero
    if consistency_weight:
        parts['cons']=_instance_variance(pf[:,1],batch['inst_h'])+_instance_variance(pf[:,0],batch['inst_v'])
    total=(parts['fwd']+.5*parts['grad']+.2*parts['page']+fold_weight*parts['fold']+
           edge_weight*parts['edge']+consistency_weight*parts['cons'])
    return total,parts


class ImprovementTracker:
    """Accumulate small improvements against a separate patience reference."""
    def __init__(self,min_delta=.02):
        self.min_delta=min_delta
        self.best=float('inf');self.reference=float('inf');self.wait=0

    def update(self,score):
        improved=score<self.best
        if improved:self.best=score
        if score<self.reference-self.min_delta:
            self.reference=score;self.wait=0
        else:self.wait+=1
        return improved
