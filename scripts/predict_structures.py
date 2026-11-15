"""Export standalone structure masks using a frozen calibration artifact.

This diagnostic does not change dewarp.py, region/keypoint export, or B/C.
Output pixels use model-input coordinates, recorded in metadata.json.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scoredewarp.learn.infer import load
from scoredewarp.learn.structure_postprocess import structure_masks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('image', type=Path)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--calibration', type=Path, required=True, help='selected.json from calibrate_structure.py')
    ap.add_argument('--method', choices=('baseline', 'threshold_only', 'threshold_ridge'), default='threshold_only')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='cpu')
    args = ap.parse_args()
    config = json.loads(args.calibration.read_text())
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if digest != config['checkpoint_sha256']:
        ap.error('Calibration belongs to a different checkpoint')
    if args.out.exists() and any(args.out.iterdir()):
        ap.error('Use a fresh empty output directory')
    image = cv2.imread(str(args.image))
    if image is None:
        ap.error('Cannot read input image')
    torch.set_num_threads(2)
    net, size = load(args.checkpoint, args.device)
    if list(size) != config['size'] or config.get('coordinate_space') != 'model input':
        ap.error('Calibration resolution/coordinate space mismatch')
    small = cv2.resize(image, tuple(size), interpolation=cv2.INTER_AREA)
    rgb = np.ascontiguousarray(small[..., ::-1].transpose(2, 0, 1), dtype=np.float32)/255
    with torch.no_grad():
        output = net(torch.from_numpy(rgb)[None].to(args.device))
        probabilities = output['struct'][0].sigmoid().cpu().numpy()
    method = config['methods'][args.method]
    masks = structure_masks(probabilities, method['thresholds'], method['radius'])
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out/'structures.npz', probabilities=probabilities, masks=masks)
    overlay = small.copy()
    for channel, (name, color) in enumerate(zip(('staff', 'jianpu_baseline', 'vertical'),
                                               ((30, 210, 30), (240, 90, 20), (30, 30, 240)))):
        if not cv2.imwrite(str(args.out/(name+'.png')), masks[channel].astype(np.uint8)*255):
            raise ValueError('Cannot write structure mask')
        overlay[masks[channel]] = color
    if not cv2.imwrite(str(args.out/'overlay.jpg'), overlay):
        raise ValueError('Cannot write overlay')
    metadata = {'image': str(args.image), 'image_sha256': hashlib.sha256(args.image.read_bytes()).hexdigest(),
                'checkpoint_sha256': digest, 'calibration_sha256': hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
                'method': args.method, 'thresholds': method['thresholds'], 'radius': method['radius'],
                'coordinate_space': 'model input', 'size': list(size), 'original_size': list(image.shape[1::-1]),
                'channels': ['staff', 'jianpu_baseline', 'vertical'],
                'deployment_ready': False, 'usage': 'standalone heatmaps; no instance/geometry confidence calibration'}
    (args.out/'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False))
    print(args.out/'metadata.json')


if __name__ == '__main__':
    main()
