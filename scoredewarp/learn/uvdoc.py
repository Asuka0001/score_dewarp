"""Adapter for the pinned official UVDoc pretrained backward control mesh.

Architecture and weights remain in data/external/uvdoc with their MIT license.
No upstream code or checkpoint is imported before its Git blob is checked.
"""
import hashlib
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

from ..core.transforms import BackwardMap

REVISION = '4c9b82b537057aff2526e6dd118a847cdd072e82'
BLOBS = {'model.py': '920e561cffa7b42331d6bb2ba27b60a5cd873c16',
         'LICENSE': 'a8a3281b35f243bd717b8681b7abe094be2f5718',
         'model/best_model.pkl': '418108b3fcc9b49df64cb3bae0f5506b3e406934'}


def load(root='data/external/uvdoc', device='cpu'):
    root = Path(root)
    for name, expected in BLOBS.items():
        path = root / name
        if not path.exists():
            raise FileNotFoundError('Run .venv/bin/python scripts/download_uvdoc.py first')
        data = path.read_bytes()
        if hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest() != expected:
            raise ValueError(f'UVDoc blob mismatch: {path}')
    spec = importlib.util.spec_from_file_location('_scoredewarp_official_uvdoc', root / 'model.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.UVDocnet(num_filter=32, kernel_size=5)
    state = torch.load(root / 'model/best_model.pkl', map_location='cpu', weights_only=True)
    model.load_state_dict(state['model_state'])
    return model.eval().to(device)


def backward_map(mesh, photo_size, out_size, stride=4):
    """Match upstream bilinear interpolation + grid_sample(align_corners=True).

    A direct output->source map needs neither a page-mask contour nor scattered
    inversion. Pixel endpoints are W-1/H-1, unlike the local forward UV model.
    """
    if mesh.ndim != 4 or mesh.shape[0:2] != (1, 2) or not torch.isfinite(mesh).all():
        raise ValueError('Expected a finite 1x2xHxW UVDoc mesh')
    w, h = photo_size
    ow, oh = out_size
    if min(w, h, ow, oh) < 2:
        raise ValueError('Image dimensions must be at least 2')
    # Sample the official continuous bilinear mesh at our actual output-grid
    # pixel coordinates, including the padded node beyond the final pixel.
    xs = np.arange(0, ow + stride, stride) / (ow - 1) * 2 - 1
    ys = np.arange(0, oh + stride, stride) / (oh - 1) * 2 - 1
    values = mesh[0].permute(1, 2, 0).detach().cpu().numpy()
    gh, gw = values.shape[:2]
    interp = RegularGridInterpolator((np.linspace(-1, 1, gh), np.linspace(-1, 1, gw)),
                                     values, bounds_error=False, fill_value=None)
    yy, xx = np.meshgrid(ys, xs, indexing='ij')
    grid = interp(np.stack([yy, xx], -1))
    grid = (grid + 1) * np.array([w - 1, h - 1]) / 2
    return BackwardMap(grid, stride, out_size)


@torch.no_grad()
def predict(model, image, out_size=None):
    h, w = image.shape[:2]
    device = next(model.parameters()).device
    rgb = cv2.cvtColor(cv2.resize(image, (488, 712)), cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float()[None].to(device) / 255
    mesh, shape3d = model(x)
    return backward_map(mesh, (w, h), out_size or (w, h)), mesh, shape3d
