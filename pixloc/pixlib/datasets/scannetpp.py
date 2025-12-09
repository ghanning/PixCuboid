import json
import logging
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import tqdm

from .base_dataset import BaseDataset, collate
from .cuboid_sampling import (
    sample_cuboid_cams, sample_cuboid_gt, sample_cuboid_random)
from .line_segments import read_line_segments
from .view import numpy_image_to_torch, read_view
from ..geometry.cuboid import Cuboid
from ..geometry import Camera, Pose
from ...settings import DATA_PATH
from ...visualization.cuboid import draw_edges

logger = logging.getLogger(__name__)


def nerfstudio_to_colmap(c2w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''Convert a camera-to-world matrix from Nerfstudio format to
    COLMAP world-to-camera (R, t).

    See https://github.com/nerfstudio-project/nerfstudio/blob/da57d3f5ba5362391b961cb4ce8b5eda4e97268f/nerfstudio/data/dataparsers/colmap_dataparser.py#L155-L160
    '''
    c2w = c2w.copy()
    c2w[2] *= -1
    c2w = c2w[[1, 0, 2, 3]]
    c2w[0:3, 1:3] *= -1
    w2c = np.linalg.inv(c2w)
    R, t = w2c[:3, :3], w2c[:3, 3]
    return R, t


class ScanNet(BaseDataset):
    default_conf = {
        'dataset_dir': 'scannetpp/',
        'image_subpath': 'data/{}/dslr/undistorted_images/',
        'transform_subpath': 'data/{}/dslr/nerfstudio/transforms_undistorted.json',
        'info_dir': 'scannetpp_pixcuboid_training/',
        'read_info_files': False,

        'train_num_per_scene': None,
        'val_num_per_scene': None,
        'test_num_per_scene': None,

        'num_views': 5,
        'init_cuboid': None,
        'init_cuboid_max_rot': float(np.deg2rad(15.0)),
        'init_cuboid_max_trans': 0.5,
        'init_cuboid_max_grow': [-0.5, 0.5],
        'init_cuboid_cam_margin': 0.5,

        'grayscale': False,
        'resize': None,
        'resize_by': 'max',
        'crop': None,
        'pad': None,
        'optimal_crop': False,
        'seed': 0,

        'max_num_points3D': 500,
        'force_num_points3D': False,

        'read_line_segments': False,
        'max_num_line_segments': 100,

        'render_edge_image': False,
        'edge_image_line_width': 3,
    }

    def _init(self, conf):
        pass

    def get_dataset(self, split):
        return _Dataset(self.conf, split)


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, conf, split):
        if conf.init_cuboid is None:
            raise ValueError('The initial cuboid sampling strategy is required.')

        self.root = Path(DATA_PATH, conf.dataset_dir)
        self.conf, self.split = conf, split

        mvc_dir = Path(__file__).parent / 'scannetpp'

        with open(mvc_dir / f'scenes_{split}.txt') as f:
            self.scenes = [line.strip() for line in f]

        with open(mvc_dir / f'layouts_{split}.json') as f:
            self.cuboids = json.load(f)

        with open(mvc_dir / f'images_{split}.json') as f:
            self.image_tuples = json.load(f)

        self.read_transform_files()

        if self.conf.read_info_files:
            self.read_info_files()

        if self.conf[self.split + '_num_per_scene']:
            self.sample_new_items(conf.seed)
        else:
            self.items = self.image_tuples

    def read_transform_files(self):
        self.transforms = {}
        self.frames = {}
        for scene in self.scenes:
            path = self.root / self.conf.transform_subpath.format(scene)
            with open(path) as f:
                transforms = json.load(f)
            self.transforms[scene] = transforms

            self.frames[scene] = {}
            for f in transforms['frames'] + transforms['test_frames']:
                self.frames[scene][f['file_path']] = f

    def read_info_files(self):
        logger.info(f'Reading info files')
        self.images, self.points3D, self.p3D_idx = {}, {}, {}
        # The below are now instead read from the transform files,
        # so that preprocessing is not required for inference.
        # self.poses, self.intrinsics, self.image_size = {}, {}, {}
        self.name2idx = {}
        for scene in tqdm.tqdm(self.scenes):
            path = Path(DATA_PATH, self.conf.info_dir, scene + '.pkl')
            with open(path, 'rb') as f:
                info = pickle.load(f)
            self.images[scene] = info['image_names']
            self.points3D[scene] = info['points3D']
            self.p3D_idx[scene] = info['p3D_idx']
            # self.poses[scene] = info['poses']
            # self.intrinsics[scene] = info['intrinsics']
            # self.image_size[scene] = info['image_size']
            self.name2idx[scene] = {name: idx for idx, name in enumerate(self.images[scene])}

    def sample_new_items(self, seed):
        logger.info(f'Sampling new images or pairs with seed {seed}')

        image_tuples_per_scene = {}
        for image_tuple in self.image_tuples:
            scene = image_tuple['scene']
            if scene not in image_tuples_per_scene:
                image_tuples_per_scene[scene] = []
            image_tuples_per_scene[scene].append(image_tuple)

        self.items = []
        num_per_scene = self.conf[self.split + '_num_per_scene']
        for scene in tqdm.tqdm(self.scenes):
            image_tuples = np.random.RandomState(seed).choice(
                image_tuples_per_scene[scene], num_per_scene, replace=False)
            self.items.extend(image_tuples)

        np.random.RandomState(seed).shuffle(self.items)

    def _read_view(self, scene, image_name, cuboid_gt, seed):
        image_dir = self.root / self.conf.image_subpath.format(scene)
        image_path = image_dir / image_name

        transforms = self.transforms[scene]
        width, height = transforms['w'], transforms['h']
        params = [transforms['fl_x'], transforms['fl_y'], transforms['cx'], transforms['cy']]
        camera = Camera.from_colmap(
            dict(model='PINHOLE', width=width, height=height, params=params)
        )

        frame = self.frames[scene][image_name]
        R, t = nerfstudio_to_colmap(np.array(frame['transform_matrix']))
        T = Pose.from_Rt(R, t)

        if self.conf.read_info_files:
            idx = self.name2idx[scene][image_name]
            p3D = self.points3D[scene]
            p3D_idx = self.p3D_idx[scene][idx]
        else:
            p3D, p3D_idx = np.empty((0, 3)), np.empty((0,), dtype=int)

        data = read_view(self.conf, image_path, camera, T, p3D, p3D_idx, random=(self.split == 'train'))
        assert tuple(data['camera'].size.numpy()) == data['image'].shape[1:][::-1]

        obs = p3D_idx
        if self.conf.crop:
            _, valid = data['camera'].world2image(data['T_w2cam']*p3D[obs])
            obs = obs[valid.numpy()]
        num_diff = self.conf.max_num_points3D - len(obs)
        if num_diff < 0:
            obs = np.random.choice(obs, self.conf.max_num_points3D, replace=False)
        num_valid_p3D = len(obs)
        if num_diff > 0 and self.conf.force_num_points3D:
            add = np.random.choice(np.delete(np.arange(len(p3D)), obs), num_diff)
            obs = np.r_[obs, add]
        data['points3D'] = torch.from_numpy(p3D[obs])
        data['p3D_mask'] = torch.zeros(data['points3D'].shape[0], dtype=bool)
        data['p3D_mask'][:num_valid_p3D] = True

        if self.conf.read_line_segments:
            lseg_path = image_dir / 'line_segments.npz'
            l2D, l2D_mask = read_line_segments(
                lseg_path, image_name, self.conf.max_num_line_segments, seed)
            data['lines2D'] = camera.normalize(l2D - 0.5).float()
            data['l2D_mask'] = torch.from_numpy(l2D_mask)

        if self.conf.render_edge_image:
            edge_image = 255.0 * np.ones(data['image'].shape[1:])
            draw_edges(cuboid_gt @ T.inv(), data['camera'], edge_image, 0.0,
                       self.conf.edge_image_line_width)
            data['edge_image'] = numpy_image_to_torch(edge_image)

        return data

    def __getitem__(self, idx):
        image_tuple = self.items[idx]
        scene = image_tuple['scene']
        seed = self.conf.seed + idx

        R, t, s = (np.array(self.cuboids[scene][key]) for key in ('R', 't', 's'))
        cuboid_gt = Cuboid.from_Rts(R, t, s)

        data = []
        for name in image_tuple['images'][:self.conf.num_views]:
            data.append(self._read_view(scene, name, cuboid_gt, seed))
        data = collate(data)

        if self.conf.init_cuboid == 'ground_truth':
            assert self.split != 'test'
            cuboid_init = sample_cuboid_gt(
                self.conf, cuboid_gt, data['T_w2cam'], seed)
        elif self.conf.init_cuboid == 'cameras':
            cuboid_init = sample_cuboid_cams(data['T_w2cam'], seed)
        elif self.conf.init_cuboid == 'random':
            cuboid_init = sample_cuboid_random(data['T_w2cam'], seed)
        else:
            raise ValueError(self.conf.init_cuboid)

        data['cuboid_init'] = cuboid_init.float()
        data['cuboid_gt'] = cuboid_gt.float()
        data['scene'] = scene
        return data

    def __len__(self):
        return len(self.items)
