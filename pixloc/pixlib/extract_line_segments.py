import argparse
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import torch
import tqdm
from deeplsd.models.deeplsd_inference import DeepLSD


def extract_line_segments(source_dir: Path, checkpoint_path: Path, min_length: float) -> Dict:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    conf = {
        'detect_lines': True,  # Whether to detect lines or only DF/AF
        'line_detection_params': {
            'merge': False,  # Whether to merge close-by lines
            'filtering': True,  # Whether to filter out lines based on the DF/AF. Use 'strict' to get an even stricter filtering
            'grad_thresh': 3,
            'grad_nfa': True,  # If True, use the image gradient and the NFA score of LSD to further threshold lines. We recommand using it for easy images, but to turn it off for challenging images (e.g. night, foggy, blurry images)
        }
    }

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    net = DeepLSD(conf)
    net.load_state_dict(ckpt['model'])
    net = net.to(device).eval()

    line_segments = {}

    for img_path in tqdm.tqdm(list(source_dir.glob('*.jpg')) + list(source_dir.glob('*.JPG'))):
        img = cv2.imread(img_path)[:, :, ::-1]
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        inputs = {'image': torch.tensor(gray_img, dtype=torch.float, device=device)[None, None] / 255.}
        with torch.no_grad():
            out = net(inputs)
            pred_lines = out['lines'][0]

        line_length = np.linalg.norm(pred_lines[:, 0] - pred_lines[:, 1], axis=1)
        mask = line_length >= min_length
        line_segments[img_path.name] = pred_lines[mask]

    return line_segments


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Line segment extraction with DeepLSD')
    parser.add_argument('--source', '-src', type=Path, required=True, help='Source directory')
    parser.add_argument('--destination', '-dst', type=Path, required=True, help='Destination .npz file')
    parser.add_argument('--checkpoint', '-ckpt', type=Path, required=True, help='Checkpoint path')
    parser.add_argument('--min_segment_length', '-msl', type=float, default=100.0, help='Minimum line segment length')
    args = parser.parse_args()

    line_segments = extract_line_segments(args.source, args.checkpoint, args.min_segment_length)
    np.savez(args.destination, **line_segments)
