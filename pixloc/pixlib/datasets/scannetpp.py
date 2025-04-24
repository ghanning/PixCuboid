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


class ScanNet(BaseDataset):
    default_conf = {
        'dataset_dir': 'scannetpp/',
        'image_subpath': 'data/{}/dslr/undistorted_images/',
        'info_dir': 'scannetpp_pixcuboid_training/',

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

        with open(mvc_dir / f'cuboids_{split}.json') as f:
            self.cuboids = json.load(f)

        with open(mvc_dir / f'images_{split}.json') as f:
            self.image_tuples = json.load(f)

        self.read_info_files()

        if self.conf[self.split + '_num_per_scene']:
            self.sample_new_items(conf.seed)
        else:
            self.add_all_items()

    def read_info_files(self):
        logger.info(f'Reading info files')
        self.images, self.points3D, self.p3D_idx = {}, {}, {}
        self.poses, self.intrinsics, self.image_size = {}, {}, {}
        self.name2idx = {}
        for scene in tqdm.tqdm(self.scenes):
            path = Path(DATA_PATH, self.conf.info_dir, scene + '.pkl')
            with open(path, 'rb') as f:
                info = pickle.load(f)
            self.images[scene] = info['image_names']
            self.points3D[scene] = info['points3D']
            self.p3D_idx[scene] = info['p3D_idx']
            self.poses[scene] = info['poses']
            self.intrinsics[scene] = info['intrinsics']
            self.image_size[scene] = info['image_size']
            self.name2idx[scene] = {name: idx for idx, name in enumerate(self.images[scene])}

    def sample_new_items(self, seed):
        logger.info(f'Sampling new images or pairs with seed {seed}')

        image_tuples_per_scene = {}
        for image_tuple in self.image_tuples:
            scene = image_tuple['scene']
            if scene not in image_tuples_per_scene:
                image_tuples_per_scene[scene] = []
            image_tuples_per_scene[scene].append(image_tuple['images'])

        self.items = []
        num_per_scene = self.conf[self.split + '_num_per_scene']
        for scene in tqdm.tqdm(self.scenes):
            image_tuples = image_tuples_per_scene[scene]
            for tuple_idx in np.random.RandomState(seed).choice(len(image_tuples), num_per_scene, replace=False):
                img_idx = [self.name2idx[scene][name] for name in image_tuples_per_scene[scene][tuple_idx]]
                self.items.append((scene, img_idx[: self.conf.num_views]))

        np.random.RandomState(seed).shuffle(self.items)

    def add_all_items(self):
        self.items = []
        for image_tuple in self.image_tuples:
            scene = image_tuple['scene']
            img_idx = [self.name2idx[scene][name] for name in image_tuple['images']]
            self.items.append((scene, img_idx[: self.conf.num_views]))

    def _read_view(self, scene, idx, cuboid_gt, seed):
        image_dir = self.root / self.conf.image_subpath.format(scene)
        image_name = self.images[scene][idx]
        image_path = image_dir / image_name

        K = self.intrinsics[scene]
        width, height = self.image_size[scene]
        camera = Camera.from_colmap(
            dict(model='PINHOLE', width=width, height=height, params=K[[0, 1, 0, 1], [0, 1, 2, 2]])
        )
        T = Pose.from_Rt(*self.poses[scene][idx])
        p3D = self.points3D[scene]
        p3D_idx = self.p3D_idx[scene][idx]
        data = read_view(self.conf, image_path, camera, T, p3D, p3D_idx, random=(self.split == 'train'))
        data['index'] = idx
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
        scene, image_idx = self.items[idx]
        seed = self.conf.seed + idx

        R, t, s = (np.array(self.cuboids[scene][key]) for key in ('R', 't', 's'))
        cuboid_gt = Cuboid.from_Rts(R, t, s)

        data = collate([self._read_view(scene, i, cuboid_gt, seed) for i in image_idx])

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
