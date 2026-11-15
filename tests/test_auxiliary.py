import unittest
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from scoredewarp.eval.auxiliary import classification_metrics, heatmap_metrics, segmentation_metrics
from scoredewarp.learn.model import ScoreDewarpNet, build_model
from scoredewarp.learn.selection import geometry_gate, auxiliary_score
from scoredewarp.learn.infer import kind_decision


class AuxiliaryTests(unittest.TestCase):
    def test_spatial_heads_roundtrip_and_preserve_geometry(self):
        parent=ScoreDewarpNet(ch=(4,8,16),coord_input=True).eval()
        net=ScoreDewarpNet(**{**parent.arch,'auxiliary_context':True}).eval()
        state=net.state_dict()
        state.update({k:v for k,v in parent.state_dict().items() if not k.startswith(('head_seg.','head_struct.'))})
        net.load_state_dict(state)
        x=torch.rand(2,3,32,48)
        for key in ('fwd','page','kind'):
            torch.testing.assert_close(net(x)[key],parent(x)[key],rtol=0,atol=0)
        self.assertEqual(net(x)['struct'].shape,(2,3,32,48))
        restored=build_model({'arch':net.arch})
        restored.load_state_dict(net.state_dict())
        self.assertTrue(restored.auxiliary_context)

    def test_untrained_unvalidated_and_low_confidence_kind_heads_fall_back(self):
        out={'kind':torch.tensor([[0.,1.,10.]])}
        net=SimpleNamespace()
        self.assertEqual(kind_decision(net,out,'staff')['page_type'],'staff')
        net.auxiliary_heads={'kind':{'trained':True,'ready':True}}
        self.assertEqual(kind_decision(net,out,'staff')['source'],'structure_detector')
        net.auxiliary_heads['kind']['deployment_geometry_passed']=True
        self.assertEqual(kind_decision(net,out,'staff')['page_type'],'mixed')
        low={'kind':torch.zeros(1,3)}
        self.assertEqual(kind_decision(net,low,'unknown')['page_type'],'unknown')
        self.assertIsNone(kind_decision(net,{'kind':torch.full((1,3),float('nan'))})['confidence'])

    def test_feature_refactor_preserves_existing_forward_exactly(self):
        net = ScoreDewarpNet(ch=(4, 8, 16), coord_input=True).eval()
        image = torch.rand(2, 3, 32, 48)
        with torch.no_grad():
            b, _, h, w = image.shape
            ys = torch.linspace(-1, 1, h).view(1, 1, h, 1).expand(b, 1, h, w)
            xs = torch.linspace(-1, 1, w).view(1, 1, 1, w).expand(b, 1, h, w)
            x, features = torch.cat([image, xs, ys], 1), []
            for i, block in enumerate(net.enc):
                x = block(x if i == 0 else F.max_pool2d(x, 2))
                features.append(x)
            kind = net.head_kind(x)
            for j, block in enumerate(net.dec):
                skip = features[-2-j]
                x = block(torch.cat([F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False), skip], 1))
            f = net.head_fwd(x)
            reference = {'fwd':f[:,:2].sigmoid(), 'page':f[:,2:3], 'kind':kind,
                         'seg':net.head_seg(x), 'struct':net.head_struct(x)}
            actual = net(image)
        for name in reference:
            torch.testing.assert_close(actual[name], reference[name], rtol=0, atol=0)

    def test_frozen_head_training_keeps_geometry_and_batchnorm_buffers(self):
        net = ScoreDewarpNet(ch=(4, 8, 16)).eval().requires_grad_(False)
        x = torch.rand(4, 3, 32, 32)
        before = net(x)['fwd'].detach().clone()
        buffers = {k:v.clone() for k,v in net.named_buffers()}
        with torch.no_grad():
            _, latent = net.forward_features(x)
            vectors = F.adaptive_avg_pool2d(latent, 1).flatten(1)
        net.head_kind.requires_grad_(True)
        old = net.head_kind[4].weight.detach().clone()
        optimizer = torch.optim.Adam(net.head_kind.parameters(), lr=.01)
        loss = F.cross_entropy(net.head_kind[2:](vectors), torch.tensor([0,1,2,0]))
        loss.backward(); optimizer.step()
        self.assertFalse(torch.equal(old,net.head_kind[4].weight))
        torch.testing.assert_close(net(x)['fwd'],before,rtol=0,atol=0)
        for k,v in net.named_buffers():
            torch.testing.assert_close(v,buffers[k],rtol=0,atol=0)

    def test_dual_domain_and_per_kind_gates_prevent_hidden_regression(self):
        reference = {'synthetic':{'ad_pct':1.,'coverage':.99},
                     'deepscores':{'ad_pct':.4,'coverage':.99},
                     'synthetic/mixed':{'jp_baseline':.1}}
        candidate = {'synthetic':{'ad_pct':.2,'coverage':.99},
                     'deepscores':{'ad_pct':.45,'coverage':.99},
                     'synthetic/mixed':{'jp_baseline':.12}}
        gate = geometry_gate(candidate,reference)
        self.assertFalse(gate['passed'])
        self.assertEqual(len(gate['reasons']),2)
        self.assertTrue(geometry_gate(reference,reference)['passed'])
        missing = geometry_gate({},reference)
        self.assertFalse(missing['passed'])
        self.assertTrue(geometry_gate({'a':{}},{'a':{'staff_straight':None}})['passed'])

    def test_coverage_and_missing_metric_gates_fail_closed(self):
        self.assertFalse(geometry_gate({'a':{'coverage':.98}}, {'a':{'coverage':.99}})['passed'])
        self.assertFalse(geometry_gate({'a':{'ad_pct':float('nan')}}, {'a':{'ad_pct':1.}})['passed'])
        self.assertFalse(geometry_gate({'a':{'ad_pct':1.}}, {'a':{'ad_pct':float('nan')}})['passed'])
        self.assertFalse(geometry_gate({}, {})['passed'])

    def test_domain_scores_have_equal_weight_and_omit_unlabeled_pixels(self):
        domains={'synthetic':{'kind':{'balanced_accuracy':.8},'seg':{'miou':.6},'struct':{'macro_f1':.4}},
                 'deepscores':{'kind':{'balanced_accuracy':1.}}}
        self.assertAlmostEqual(auxiliary_score(domains,'kind'),.9)
        self.assertAlmostEqual(auxiliary_score(domains,'pixels'),.5)

    def test_unknown_types_and_absent_classes_are_not_perfect_scores(self):
        m=classification_metrics([0,0],[0,3])
        self.assertEqual(m['accuracy'],.5)
        self.assertEqual(m['balanced_accuracy'],.5)
        self.assertIsNone(m['recall'][1])
        self.assertIsNone(classification_metrics([],[])['accuracy'])
        self.assertIsNone(segmentation_metrics(np.zeros((5,5)))['miou'])
        self.assertIsNone(heatmap_metrics([0,0,0],[0,0,0],[0,0,0])['macro_f1'])
        self.assertEqual(heatmap_metrics([0],[4],[0])['macro_f1'],0.)


if __name__=='__main__':
    unittest.main()
