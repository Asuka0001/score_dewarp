import unittest
import numpy as np
import torch

from scoredewarp.core.structures import Fragment, Structures, group_staves
from scoredewarp.core.transforms import BackwardMap
from scoredewarp.eval.geometry import (geometry_metrics, map_points, network_points,
                                       paired_geometry_metrics)
from scoredewarp.eval.metrics import evaluate
from scoredewarp.eval.structures import staff_detection_metrics


def staff_gt():
    ann = []
    for i in range(5):
        x = np.linspace(20, 180, 25)
        ann.append({'kind':'staff_line','staff':0,'line':i,
                    'pts':np.stack([x, np.full_like(x, 40 + 12*i)], 1).tolist()})
    return {'meta':{'size':[200,240]},'flat_ann':ann,'photo_ann':ann}


class StructureMetricsTests(unittest.TestCase):
    def test_common_points_reveal_error_hidden_by_cropping(self):
        y,x = np.mgrid[:12,:12]
        target = np.stack([x,y],-1).astype(float)
        valid = np.ones((12,12),bool)
        pred = target.copy(); pred[:,6:,1] += 4
        keep = valid.copy(); keep[:,6:] = False
        own = geometry_metrics(target,pred,valid)
        pair = paired_geometry_metrics(target,{'cropped':(target,keep),'full':(pred,valid)},valid)
        self.assertGreater(own['ad_pct'], 1)
        self.assertEqual(pair['full']['coverage'],.5)
        self.assertAlmostEqual(pair['full']['ad_pct'],0)
        self.assertEqual(pair['full']['coverage'],pair['cropped']['coverage'])

    def test_empty_intersection_is_undefined_not_perfect(self):
        y,x=np.mgrid[:8,:8]; p=np.stack([x,y],-1).astype(float)
        left=x<4
        pair=paired_geometry_metrics(p,{'a':(p,left),'b':(p,~left)},np.ones(x.shape,bool))
        for m in pair.values():
            self.assertEqual(m['coverage'],0)
            self.assertIsNone(m['ad_pct'])

    def test_map_checks_reject_canvas_and_roundtrip_holes(self):
        bm=BackwardMap.from_homography(np.eye(3),(80,80))
        p=np.array([[20.,20.],[81.,20.],[-3.,20.]])
        _,keep=map_points(bm,p)
        np.testing.assert_array_equal(keep,[True,False,False])
        bm.sample=lambda pts: np.zeros_like(pts)
        _,keep=map_points(bm,p)
        self.assertFalse(keep.any())

    def test_network_mask_sampling_preserves_point_and_grid_shapes(self):
        y,x=np.mgrid[:16,:16]
        out={'fwd':torch.tensor(np.stack([x/16,y/16]),dtype=torch.float32)[None],
             'page':torch.full((1,1,16,16),10.)}
        for shape in ((8,2),(2,4,2)):
            p=np.full(shape,32.)
            coords,page=network_points(out,p,(64,64),1.)
            self.assertEqual(coords.shape,p.shape)
            self.assertEqual(page.shape,p.shape[:-1])
            np.testing.assert_allclose(coords,.5)

    def test_staff_metrics_and_annotation_coverage_use_shared_support(self):
        gt=staff_gt()
        identity=lambda p:p
        m=evaluate(gt,identity)
        self.assertAlmostEqual(m['staff_straight'],0)
        self.assertAlmostEqual(m['staff_spacing_cv'],0)
        crop=evaluate(gt,identity,point_valid=lambda p:p[:,0]<100)
        self.assertLess(crop['staff_line_coverage'],.5)
        self.assertLess(crop['keypoints_valid'],crop['keypoints_total'])
        missing=evaluate(gt,identity,point_valid=lambda p:np.zeros(len(p),bool))
        self.assertEqual(missing['coverage'],0)
        self.assertNotIn('staff_straight',missing)
        outside=evaluate(gt,lambda p:p+300,out_size=(200,240))
        self.assertEqual(outside['coverage'],0)

    def test_missing_annotations_and_missing_jianpu_do_not_fake_scores(self):
        self.assertEqual(evaluate({'meta':{'size':[200,240]},'flat_ann':[],'photo_ann':[]},lambda p:p),{})
        ann=[]
        for i in range(6):
            p=[[20*i,20],[20*i+10,20],[20*i+10,40],[20*i,40],
               [20*i+5,20],[20*i+5,40],[20*i+5,30]]
            ann.append({'kind':'jp_char','row':0,'pts':p})
        gt={'meta':{'size':[200,240]},'flat_ann':ann,'photo_ann':ann}
        m=evaluate(gt,lambda p:np.full_like(p,np.nan))
        self.assertEqual(m['jp_char_coverage'],0)
        self.assertNotIn('jp_baseline',m)

    def test_staff_detection_matching_penalizes_duplicate_and_missing_systems(self):
        gt=staff_gt()
        fs=[Fragment(np.asarray(a['pts'])) for a in gt['flat_ann']]
        ss=group_staves(fs,200,12)
        structures=Structures((240,200),2,12,fs,ss,[],[])
        m=staff_detection_metrics(gt,lambda p:p,structures)
        self.assertEqual(m['staff_precision'],1)
        self.assertEqual(m['staff_recall'],1)
        structures.staves=ss+ss
        m=staff_detection_metrics(gt,lambda p:p,structures)
        self.assertEqual(m['staff_precision'],.5)
        self.assertEqual(m['staff_recall'],1)
        structures.staves=[]
        m=staff_detection_metrics(gt,lambda p:p,structures)
        self.assertEqual(m['staff_recall'],0)
        self.assertIsNone(m['staff_precision'])


if __name__=='__main__':
    unittest.main()
