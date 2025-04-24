from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def read_line_segments(
    path: Path, name: str, max_num_segments: int, seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    '''Read line segments from .npz file.

    Args:
        path: path to .npz file.
        name: image name.
        max_num_segments: maximum number of segments to return.
        seed: random seed.
    Returns:
        line_segments: the line segments (N, 2, 2).
        mask: boolean array indicating whether segments are valid (N).
    '''
    data = np.load(path)
    line_segments = data[name]

    if line_segments.shape[0] > max_num_segments:
        idx = np.random.RandomState(seed).choice(
            line_segments.shape[0], max_num_segments, replace=False)
        line_segments = line_segments[idx]

    mask = np.ones(line_segments.shape[0], dtype=bool)
    num_pad = max_num_segments - line_segments.shape[0]
    line_segments = np.pad(line_segments, ((0, num_pad), (0, 0), (0, 0)))
    mask = np.pad(mask, (0, num_pad))

    return line_segments, mask
