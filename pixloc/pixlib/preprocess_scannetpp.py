from multiprocessing import Pool
from pathlib import Path
import functools
import json
import pickle
from typing import Dict, List, Tuple

import cv2
import numpy as np
import scipy
import tqdm
from plyfile import PlyData

from .geometry.wrappers import Camera, Pose
from .preprocess_megadepth import assemble_intrinsics
from ..settings import DATA_PATH
from ..utils.colmap import read_model
from .. import logger

DEPTH_MAP_INVALID = 0
DEPTH_MAP_SCALE = 0.001


def read_mesh(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    mesh = PlyData.read(path, known_list_len={'face': {'vertex_indices': 3}})
    verts = np.stack([mesh['vertex']['x'], mesh['vertex']['y'], mesh['vertex']['z']]).T
    labels = np.array(mesh['vertex']['label'])
    return verts, labels


def read_classes(path: Path) -> Dict:
    with open(path) as f:
        classes = [line.strip() for line in f]
    class2idx = {class_: idx for idx, class_ in enumerate(classes)}
    return class2idx


def sample_depth_map(depth_map: np.ndarray, p2D: np.ndarray) -> np.ndarray:
    idx = np.rint(p2D).astype(int)
    return depth_map[idx[:, 1], idx[:, 0]]


def farthest_point_sampling(p: np.ndarray, dist_thr: float) -> np.ndarray:
    mask = np.zeros(p.shape[0], dtype=bool)

    if p.shape[0] == 0:
        return mask

    sqr_dist_thr = dist_thr * dist_thr
    min_sqr_dist = np.full(p.shape[0], np.inf)
    idx = 0

    while True:
        mask[idx] = True
        sqr_dist = scipy.spatial.distance.cdist(p[None, idx], p, 'sqeuclidean')
        min_sqr_dist = np.minimum(min_sqr_dist, sqr_dist[0])
        idx = np.argmax(min_sqr_dist)
        if min_sqr_dist[idx] < sqr_dist_thr:
            break

    return mask


def find_points_fwc(
    verts: np.ndarray,
    vert_labels: np.ndarray,
    class2label: Dict,
    images: Dict,
    image_ids: List[int],
    camera: Camera,
    depth_map_dir: Path,
) -> List[np.ndarray]:
    vert_mask = np.zeros(verts.shape[0], dtype=bool)
    for class_ in ('floor', 'wall', 'ceiling'):
        vert_mask[vert_labels == class2label[class_]] = True
    vert_idx = np.where(vert_mask)[0]

    max_num_verts = 100000
    if vert_idx.shape[0] > max_num_verts:
        vert_idx = np.random.RandomState(0).choice(vert_idx, max_num_verts, replace=False)

    p3D_idx = []
    for image_id in tqdm.tqdm(image_ids):
        image = images[image_id]
        T_w2cam = Pose.from_colmap(image)

        idx = vert_idx
        p3D = T_w2cam * verts[idx]
        p2D, valid = camera.world2image(p3D)
        p2D, idx, z = p2D[valid].numpy(), idx[valid], p3D[valid, 2].numpy()

        depth_map = cv2.imread(depth_map_dir / image.name.replace('JPG', 'png'), cv2.IMREAD_ANYDEPTH)
        depth = sample_depth_map(depth_map, p2D)
        ok = (depth != DEPTH_MAP_INVALID) & (z < DEPTH_MAP_SCALE * depth + 0.1)
        p2D, idx = p2D[ok], idx[ok]

        keep = farthest_point_sampling(p2D, 8.0)
        p2D, idx = p2D[keep], idx[keep]

        p3D_idx.append(idx.astype(np.int32))

    return p3D_idx


def preprocess_scene(scene, root):
    logger.info(f'Preprocessing scene {scene}.')

    scene_dir = root / 'data' / scene
    device = 'dslr'

    colmap_dir = scene_dir / device / 'colmap'
    cameras, images, points3D = read_model(colmap_dir)
    image_ids = sorted(images.keys())
    logger.info(
        f'Read COLMAP model with {len(cameras)} camera(s), {len(images)} image(s) and {len(points3D)} 3D point(s).'
    )

    mesh_path = scene_dir / 'scans' / 'mesh_aligned_0.05_semantic.ply'
    verts, vert_labels = read_mesh(mesh_path)
    assert verts.shape[0] == vert_labels.shape[0]
    logger.info(f'Read mesh with {verts.shape[0]} vertices.')

    classes_path = root / 'metadata' / 'semantic_classes.txt'
    class2label = read_classes(classes_path)
    logger.info(f'{len(class2label)} semantic classes.')

    transforms_path = scene_dir / device / 'nerfstudio' / 'transforms_undistorted.json'
    with open(transforms_path) as f:
        transforms = json.load(f)
    assert transforms['camera_model'] == 'PINHOLE'
    params = transforms['fl_x'], transforms['fl_y'], transforms['cx'], transforms['cy']
    K = assemble_intrinsics(*params)

    camera = Camera.from_colmap(
        {'model': 'PINHOLE', 'params': params, 'width': transforms['w'], 'height': transforms['h']}
    )
    depth_map_dir = scene_dir / device / 'render_depth_undistorted'
    p3D_idx = find_points_fwc(verts, vert_labels, class2label, images, image_ids, camera, depth_map_dir)

    data = {
        'points3D': verts.astype(np.float32),
        'p3D_idx': p3D_idx,
        'intrinsics': K,
        'image_size': (transforms['w'], transforms['h']),
        'poses': [(images[id].qvec2rotmat(), images[id].tvec) for id in image_ids],
        'image_names': [images[id].name for id in image_ids],
    }
    return data


def preprocess_and_write(scene, root, out_dir, **kwargs):
    path = out_dir / (scene + '.pkl')
    if path.exists():
        return

    try:
        data = preprocess_scene(scene, root, **kwargs)
    except:  # noqa  E722
        logger.info(f'Error for scene {scene}.')
        raise
    if data is None:
        return

    logger.info(f'Writing scene {scene} to {path}.')
    with open(path, 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    root = DATA_PATH / 'scannetpp/'
    out_dir = DATA_PATH / 'scannetpp_pixcuboid_training/'
    out_dir.mkdir(exist_ok=True)

    scenes = []
    for split in ('train', 'val', 'test'):
        split_path = Path(__file__).parent / 'datasets' / 'scannetpp' / f'scenes_{split}.txt'
        with open(split_path) as f:
            scenes.extend([line.strip() for line in f])

    logger.info(f'Found {len(scenes)} scenes.')

    fn = functools.partial(preprocess_and_write, root=root, out_dir=out_dir)
    with Pool(5) as p:
        p.map(fn, scenes)
