import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch import Tensor

from . import Pose, Camera
from .check_jacobians import compute_J, print_J_diff
from .cuboid import Cuboid
from .cuboid_costs import FeaturemetricCost, EdgeCost, VanishingPointCost
from .interpolation import Interpolator
from .optimization import so3exp_map


def apply_delta(cuboid: Cuboid, delta: Tensor):
    dw, dd = delta.split([3, 6])
    dR = so3exp_map(dw)
    return Cuboid.from_Rd(dR @ cuboid.R, cuboid.d + dd)


def toy_problem(seed=0, n_cams=8, n_points=128, n_lseg=64):
    torch.random.manual_seed(seed)
    state = np.random.RandomState(seed=seed)

    R = Rotation.random(random_state=state).as_matrix()
    d = torch.cat([-2.0 - torch.rand(3), 2.0 + torch.rand(3)])
    cuboid = Cuboid.from_Rd(R, d)

    T_w2cam = []
    for _ in range(n_cams):
        R = Rotation.random(random_state=state).as_matrix()
        t = -1.0 + 2.0 * torch.rand(3, dtype=torch.float64)
        T_w2cam.append(Pose.from_Rt(R, t))
    T_w2cam = torch.stack(T_w2cam)

    w, h = 640, 480
    size = torch.tensor([w, h])
    fx, fy = 300., 350.
    cx, cy = w/2, h/2
    params = torch.tensor([w, h, fx, fy, cx, cy])
    camera = torch.stack([Camera(params) for _ in range(n_cams)]).double()

    p2D = torch.rand(n_cams, n_points, 2, dtype=torch.float64) * size

    dim = 16
    F = torch.randn(n_cams, dim, h, w, dtype=torch.float64)

    E = torch.randn(n_cams, 1, h, w, dtype=torch.float64)

    l2D = torch.rand(n_cams, n_lseg, 2, 2, dtype=torch.float64) * size

    return cuboid, T_w2cam, camera, p2D, F, E, l2D


def test_J_sample_edges(cuboid: Cuboid, num_points: int = 10):
    p3D, J = cuboid.sample_edges(num_points, return_jacobian=True)
    delta = torch.zeros(9).to(p3D)
    fn = lambda d: apply_delta(cuboid, d).sample_edges(num_points)[0]
    J_auto = compute_J(fn, delta)

    print_J_diff('sample edges', J, J_auto)
    torch.testing.assert_close(J, J_auto)


def test_J_project(cuboid: Cuboid, p2D: Tensor, T_w2cam: Pose):
    p3D, J = cuboid.project(p2D, T_w2cam, return_jacobian=True)
    delta = torch.zeros(9).to(p3D)
    fn = lambda d: apply_delta(cuboid, d).project(p2D, T_w2cam)[0]
    J_auto = compute_J(fn, delta)

    print_J_diff('project', J, J_auto)
    torch.testing.assert_close(J, J_auto)


def test_J_featuremetric_cost(
        cuboid: Cuboid, T_w2cam: Pose, camera: Camera, p2D: Tensor, F: Tensor):

    interpolator = Interpolator(mode='cubic', pad=2)
    cost = FeaturemetricCost(interpolator)

    F_interp, _, _ = interpolator(F, p2D)
    p2D = camera.normalize(p2D)

    args = (T_w2cam, camera, p2D, F, F_interp)
    res, valid, weight, info = cost.residuals(cuboid, *args, do_gradients=True)
    J = cost.jacobian(T_w2cam, camera, *info)

    delta = torch.zeros(9).to(p2D)
    fn = lambda d: cost.residuals(apply_delta(cuboid, d), *args)[0]
    J_auto = compute_J(fn, delta)

    J, J_auto = J[valid], J_auto[valid]
    print_J_diff('featuremetric cost', J, J_auto)
    torch.testing.assert_close(J, J_auto)


def test_J_edge_cost(
        cuboid: Cuboid, T_w2cam: Pose, camera: Camera, E: torch.Tensor):

    interpolator = Interpolator(mode='cubic', pad=2)
    cost = EdgeCost(interpolator, num_points=20)

    args = (T_w2cam, camera, E)
    res, valid, weight, info = cost.residuals(cuboid, *args, do_gradients=True)
    J = cost.jacobian(T_w2cam, camera, *info)

    delta = torch.zeros(9).to(E)
    fn = lambda d: cost.residuals(apply_delta(cuboid, d), *args)[0]
    J_auto = compute_J(fn, delta)

    J, J_auto = J[valid], J_auto[valid]
    print_J_diff('edge cost', J, J_auto)
    torch.testing.assert_close(J, J_auto)


def test_J_vp_cost(cuboid: Cuboid, T_w2cam: Pose, l2D: Tensor):
    cost = VanishingPointCost(dist_thr=0.05)

    res, info = cost.residuals(cuboid, T_w2cam, l2D)
    J = cost.jacobian(l2D, *info)

    delta = torch.zeros(9).to(res)
    fn = lambda d: cost.residuals(apply_delta(cuboid, d), T_w2cam, l2D)[0]
    J_auto = compute_J(fn, delta)

    print_J_diff('vanishing point cost', J, J_auto)
    torch.testing.assert_close(J, J_auto)


def main():
    cuboid, T_w2cam, camera, p2D, F, E, l2D = toy_problem()
    test_J_sample_edges(cuboid)
    test_J_project(cuboid, camera[0].normalize(p2D[0]), T_w2cam[0])

    few_points = 6
    test_J_featuremetric_cost(cuboid, T_w2cam, camera, p2D[:, :few_points], F)
    test_J_edge_cost(cuboid, T_w2cam, camera, E)
    test_J_vp_cost(cuboid, T_w2cam, camera.unsqueeze(-1).normalize(l2D))


if __name__ == '__main__':
    main()
