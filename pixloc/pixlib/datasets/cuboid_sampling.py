from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from ..geometry import Pose
from ..geometry.cuboid import Cuboid


def orthogonal_vector(x: np.ndarray) -> np.ndarray:
    y = np.zeros_like(x)
    m = (x != 0).argmax()
    n = m + 1 if m + 1 < x.shape[0] else 0
    y[n] = x[m]
    y[m] = -x[n]
    return y


def sample_cuboid_cams(
    T_w2cam: Pose, seed: Optional[int] = None, margin: float = 2.5
) -> Cuboid:
    '''Sample cuboid based on known camera poses.

    Args:
        T_w2cam: camera poses.
        seed: random seed.
        margin: distance between camera centers and cuboid faces.
    Returns:
        cuboid: sampled cuboid.
    '''
    state = np.random.RandomState(seed)

    R_cam, t_cam = T_w2cam.numpy()
    cams_w = -R_cam.transpose(0, 2, 1) @ t_cam[..., None]

    down = np.mean(R_cam[:, 1], axis=0)
    z = -down
    z /= np.linalg.norm(z)
    y = orthogonal_vector(z)
    y /= np.linalg.norm(y)
    x = np.cross(y, z)

    a = state.uniform(0, 2 * np.pi)
    y = np.sin(a) * y + np.cos(a) * x
    x = np.cross(y, z)
    R = np.stack([x, y, z])

    cams = cams_w[..., 0] @ R.T
    d = np.r_[np.min(cams, axis=0) - margin, np.max(cams, axis=0) + margin]

    return Cuboid.from_Rd(R, d)


def sample_cuboid_random(
    T_w2cam: Pose, seed: Optional[int] = None, margin: float = 2.5
) -> Cuboid:
    '''Sample cuboid with random orientation.

    Args:
        T_w2cam: camera poses.
        seed: random seed.
        margin: distance between camera centers and cuboid faces.
    Returns:
        cuboid: sampled cuboid.
    '''
    state = np.random.RandomState(seed)
    R = Rotation.random(random_state=state).as_matrix()

    R_cam, t_cam = T_w2cam.numpy()
    cams_w = -R_cam.transpose(0, 2, 1) @ t_cam[..., None]

    cams = cams_w[..., 0] @ R.T
    d = np.r_[np.min(cams, axis=0) - margin, np.max(cams, axis=0) + margin]

    return Cuboid.from_Rd(R, d)


def sample_cuboid_gt(
    conf, cuboid_gt: Cuboid, T_w2cam: Pose, seed: Optional[int] = None,
) -> Cuboid:
    '''Sample cuboid by perturbing a ground truth cuboid.

    Args:
        conf: config.
        cuboid_gt: the ground truth cuboid.
        T_w2cam: camera poses.
        seed: random seed.
    Returns:
        cuboid: sampled cuboid.
    '''
    state = np.random.RandomState(seed)

    # https://math.stackexchange.com/questions/1585975/how-to-generate-random-points-on-a-sphere
    axis = state.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = state.uniform(0, conf.init_cuboid_max_rot)
    dR = Rotation.from_rotvec(angle * axis).as_matrix()
    R = cuboid_gt.R @ dR

    dt = state.uniform(
        -conf.init_cuboid_max_trans, conf.init_cuboid_max_trans, 3)
    t = cuboid_gt.t + dt

    ds = state.uniform(
        conf.init_cuboid_max_grow[0], conf.init_cuboid_max_grow[1], 3)
    s = cuboid_gt.s + ds

    cuboid = Cuboid.from_Rts(R, t, s)

    R_cam, t_cam = T_w2cam.numpy()
    cams_w = -R_cam.transpose(0, 2, 1) @ t_cam[..., None]
    return cuboid.resize(cams_w.squeeze(-1), True, conf.init_cuboid_cam_margin)
