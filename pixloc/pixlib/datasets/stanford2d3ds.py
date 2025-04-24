import json
from pathlib import Path

import numpy as np
import torch

from .base_dataset import BaseDataset, collate
from .cuboid_sampling import sample_cuboid_cams
from .line_segments import read_line_segments
from .view import read_view
from ..geometry import Camera, Pose
from ..geometry.cuboid import Cuboid
from ...settings import DATA_PATH


class Stanford2D3DS(BaseDataset):
    default_conf = {
        'dataset_dir': '2d3ds/',
        'image_subpath': '{}/persp/rgb/',
        'pose_subpath': '{}/persp/pose/',

        'grayscale': False,
        'resize': None,
        'resize_by': 'max',
        'crop': None,
        'pad': None,
        'optimal_crop': False,
        'seed': 0,

        'read_line_segments': False,
        'max_num_line_segments': 100,
    }

    def _init(self, conf):
        pass

    def get_dataset(self, split):
        return _Dataset(self.conf, split)


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, conf, split):
        self.root = Path(DATA_PATH, conf.dataset_dir)
        self.conf = conf

        mvc_dir = Path(__file__).parent / '2d3ds'

        with open(mvc_dir / 'cuboids_all.json') as f:
            self.cuboids = json.load(f)

        with open(mvc_dir / 'images_all.json') as f:
            self.image_tuples = json.load(f)

    def _read_view(self, area, image_name, seed):
        image_dir = self.root / self.conf.image_subpath.format(area)
        image_path = image_dir / image_name

        pose_dir = self.root / self.conf.pose_subpath.format(area)
        pose_name = image_name.replace('rgb', 'pose').replace('.jpg', '.json')
        pose_path = pose_dir / pose_name
        with open(pose_path) as f:
            pose = json.load(f)

        K = np.array(pose['camera_k_matrix'])
        params = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
        camera = Camera.from_colmap(dict(
            model='PINHOLE', params=params,
            width=pose['image_width'], height=pose['image_height']))

        Rt = np.array(pose['camera_rt_matrix'])
        T = Pose.from_Rt(Rt[:3, :3], Rt[:, 3])

        data = read_view(
            self.conf, image_path, camera, T, np.empty((0, 3)), np.empty(0))

        if self.conf.read_line_segments:
            lseg_path = image_dir / 'line_segments.npz'
            l2D, l2D_mask = read_line_segments(
                lseg_path, image_name, self.conf.max_num_line_segments, seed)
            data['lines2D'] = camera.normalize(l2D - 0.5).float()
            data['l2D_mask'] = torch.from_numpy(l2D_mask)

        return data

    def __getitem__(self, idx):
        image_tuple = self.image_tuples[idx]
        seed = self.conf.seed + idx

        scene = image_tuple['scene']
        area, space = scene.split(':')

        data = collate([self._read_view(area, i, seed) for i in image_tuple['perspective_images']])

        cuboid_gt = self.cuboids[scene]
        R, t, s = (np.array(cuboid_gt[key]) for key in ('R', 't', 's'))
        data['cuboid_gt'] = Cuboid.from_Rts(R, t, s).float()

        cuboid_init = sample_cuboid_cams(data['T_w2cam'], seed)
        data['cuboid_init'] = cuboid_init.float()

        data['area'] = area
        data['space'] = space
        return data

    def __len__(self):
        return len(self.image_tuples)
