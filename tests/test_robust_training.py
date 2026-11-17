import unittest

import numpy as np
import torch

from scoredewarp.learn.robust_geometry import ImprovementTracker,geometry_loss
from scoredewarp.learn.diverse_dataset import augment
from scoredewarp.synth.distort import PageWarp,projection_diagnostics


class RobustTrainingTests(unittest.TestCase):
    def batch(self):
        y,x=torch.meshgrid(torch.linspace(0,1,20),torch.linspace(0,1,24),indexing='ij')
        target=torch.stack([x,y])[None]
        return {'fwd':target,'page':torch.ones(1,1,20,24),
                'inst_h':torch.zeros(1,20,24,dtype=torch.long),'inst_v':torch.zeros(1,20,24,dtype=torch.long)}

    def test_small_improvements_accumulate_for_patience(self):
        tracker=ImprovementTracker(.02)
        for score in (1.,.991,.982):self.assertTrue(tracker.update(score))
        self.assertEqual(tracker.wait,2)
        self.assertTrue(tracker.update(.973))
        self.assertEqual(tracker.wait,0)
        self.assertFalse(tracker.update(.974))
        self.assertEqual(tracker.wait,1)

    def test_fold_penalty_detects_reversal_and_has_finite_gradient(self):
        batch=self.batch();good=batch['fwd'].clone().requires_grad_()
        _,parts=geometry_loss({'fwd':good,'page':torch.zeros_like(batch['page'])},batch)
        self.assertEqual(parts['fold'].item(),0.)
        bad=batch['fwd'].flip(-1).clone().requires_grad_()
        total,parts=geometry_loss({'fwd':bad,'page':torch.zeros_like(batch['page'])},batch)
        self.assertGreater(parts['fold'].item(),.5)
        total.backward()
        self.assertTrue(torch.isfinite(bad.grad).all())
        self.assertGreater(bad.grad.abs().sum(),0)

    def test_edge_loss_includes_image_boundary_on_full_page(self):
        batch=self.batch();pred=batch['fwd'].clone();pred[:,:,:,0]+=.1
        _,parts=geometry_loss({'fwd':pred,'page':torch.zeros_like(batch['page'])},batch)
        self.assertGreater(parts['edge'].item(),0)
        empty={**batch,'page':torch.zeros_like(batch['page'])}
        total,parts=geometry_loss({'fwd':pred,'page':torch.zeros_like(batch['page'])},empty)
        self.assertTrue(torch.isfinite(total))
        self.assertEqual(parts['fold'].item(),0.)

    def test_fill_defaults_preserve_old_warp_and_tight_crop_remains_invertible(self):
        a=PageWarp((1240,1754),(768,1024),np.random.default_rng(7))
        b=PageWarp((1240,1754),(768,1024),np.random.default_rng(7),fill_range=(.72,.93))
        pts=np.array([[0.,0.],[1240.,1754.],[600.,900.]])
        np.testing.assert_array_equal(a.forward(pts),b.forward(pts))
        tight=PageWarp((1240,1754),(768,1024),np.random.default_rng(7),strength=0,fill_range=(1.2,1.3))
        self.assertTrue(projection_diagnostics(tight)['valid'])
        outline=tight.forward(pts[:2])
        self.assertTrue((outline[0]<0).any() or (outline[1]>[768,1024]).any())

    def test_online_crops_preserve_planar_geometry_and_are_seed_reproducible(self):
        batch=self.batch()
        sample={k:v[0].numpy() for k,v in batch.items()}
        sample['image']=np.ones((3,20,24),np.float32)*.8
        original={k:v.copy() for k,v in sample.items()}
        for seed in range(10):
            transformed=augment(sample,seed)
            for key in transformed:np.testing.assert_array_equal(transformed[key],augment(sample,seed)[key])
            field=transformed['fwd'].transpose(1,2,0)
            dx=field[:-1,1:]-field[:-1,:-1];dy=field[1:,:-1]-field[:-1,:-1]
            self.assertTrue(((dx[...,0]*dy[...,1]-dx[...,1]*dy[...,0])>0).all())
            self.assertTrue(np.isfinite(transformed['image']).all())
        for key in sample:np.testing.assert_array_equal(sample[key],original[key])


if __name__=='__main__':unittest.main()
