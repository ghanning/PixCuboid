import torch
from typing import Optional
from torch import Tensor

from . import Pose, Camera
from .cuboid import Cuboid
from .optimization import J_normalization, skew_symmetric
from .interpolation import Interpolator
from .utils import to_homogeneous


class FeaturemetricCost:
    def __init__(self, interpolator: Interpolator, normalize: bool = False):
        self.interpolator = interpolator
        self.normalize = normalize

    def residuals(
            self, cuboid: Cuboid, T_w2cam: Pose, camera: Camera,
            p2D_inp: Tensor, F: Tensor, F_interp: Tensor,
            W: Optional[Tensor] = None, W_interp: Optional[Tensor] = None,
            do_gradients: bool = False):

        p3D_w, J_p3Dw_Rd = cuboid.unsqueeze(-1).project(
            p2D_inp, T_w2cam, return_jacobian=do_gradients)
        p3D_w = p3D_w.flatten(start_dim=-3, end_dim=-2).unsqueeze(-3)

        p3D_cam = T_w2cam * p3D_w
        p2D, visible = camera.world2image(p3D_cam)

        F_p2D_raw, valid, gradients = self.interpolator(
            F, p2D, return_gradients=do_gradients)
        valid = valid & visible

        # Don't project back into the same camera
        num_cam, num_pt = p2D_inp.shape[-3:-1]
        mask = ~torch.eye(num_cam, dtype=bool, device=cuboid.device)
        mask = mask.repeat(cuboid.shape + (1, 1))
        # Don't warp from or to cameras outside the cuboid
        cams_w = (-T_w2cam.t.unsqueeze(-2) @ T_w2cam.R).squeeze(-2)
        outside = ~cuboid.inside(cams_w)
        mask.masked_fill_(outside.unsqueeze(-1), False)
        mask.masked_fill_(outside.unsqueeze(-2), False)
        mask = mask.repeat_interleave(num_pt, -1)
        valid = valid & mask

        if W is not None:
            W_p2D, _, _ = self.interpolator(
                W, p2D, return_gradients=False)
            weight = W_interp.repeat(1, 1, num_cam, 1) * W_p2D
            weight = weight.squeeze(-1).masked_fill(~valid, 0.)
        else:
            weight = None

        if self.normalize:
            F_p2D = torch.nn.functional.normalize(F_p2D_raw, dim=-1)
        else:
            F_p2D = F_p2D_raw

        res = F_p2D - F_interp.flatten(-3, -2).unsqueeze(-3)
        info = (p3D_cam, F_p2D_raw, gradients, J_p3Dw_Rd)
        return res, valid, weight, info

    def jacobian(
            self, T_w2cam: Pose, camera: Camera, p3D_cam: Tensor,
            F_p2D_raw: Tensor, J_f_p2D: Tensor, J_p3Dw_Rd: Tensor):

        J_p3Dcam_p3Dw = T_w2cam.R.unsqueeze(-3)
        J_p3Dcam_Rd = J_p3Dcam_p3Dw @ J_p3Dw_Rd.flatten(-4, -3).unsqueeze(-4)

        J_p2D_p3Dcam, _ = camera.J_world2image(p3D_cam)

        if self.normalize:
            J_f_p2D = J_normalization(F_p2D_raw) @ J_f_p2D

        J_p2D_Rd = J_p2D_p3Dcam @ J_p3Dcam_Rd
        J = J_f_p2D @ J_p2D_Rd
        return J

    def residual_jacobian(
            self, cuboid: Cuboid, T_w2cam: Pose, camera: Camera,
            p2D: Tensor, F: Tensor, F_interp: Tensor,
            W: Optional[Tensor] = None, W_interp: Optional[Tensor] = None):

        res, valid, weight, info = self.residuals(
            cuboid, T_w2cam, camera, p2D, F, F_interp, W, W_interp, True)
        J = self.jacobian(T_w2cam, camera, *info)
        return res, valid, weight, J


class EdgeCost:
    def __init__(self, interpolator: Interpolator, num_points: int):
        self.interpolator = interpolator
        self.num_points = num_points

    def residuals(
            self, cuboid: Cuboid, T_w2cam: Pose, camera: Camera,
            E: Tensor, W: Optional[Tensor] = None,
            do_gradients: bool = False):

        p3D_w, J_p3Dw_Rd = cuboid.unsqueeze(-1).sample_edges(
            self.num_points, return_jacobian=do_gradients)
        p3D_cam = T_w2cam * p3D_w

        p2D, visible = camera.world2image(p3D_cam)
        E_p2D, valid, J_e_p2D = self.interpolator(
            E, p2D, return_gradients=do_gradients)
        valid = valid & visible

        cams_w = (-T_w2cam.t.unsqueeze(-2) @ T_w2cam.R).squeeze(-2)
        inside = cuboid.inside(cams_w)
        valid = valid & inside.unsqueeze(-1)

        if W is not None:
            W_p2D, _, _ = self.interpolator(W, p2D)
            weight = W_p2D.squeeze(-1).masked_fill(~valid, 0.)
        else:
            weight = None

        info = (p3D_cam, J_p3Dw_Rd, J_e_p2D)
        return E_p2D, valid, weight, info

    def jacobian(
            self, T_w2cam: Pose, camera: Camera, p3D_cam: Tensor,
            J_p3Dw_Rd: Tensor, J_e_p2D: Tensor):

        J_p3D_p3Dw = T_w2cam.R.unsqueeze(-3)
        J_p2D_p3D, _ = camera.J_world2image(p3D_cam)
        J = J_e_p2D @ J_p2D_p3D @ J_p3D_p3Dw @ J_p3Dw_Rd
        return J

    def residual_jacobian(
            self, cuboid: Cuboid, T_w2cam: Pose, camera: Camera,
            E: Tensor, W: Optional[Tensor] = None):

        res, valid, weight, info = self.residuals(
            cuboid, T_w2cam, camera, E, W, True)
        J = self.jacobian(T_w2cam, camera, *info)
        return res, valid, weight, J


class VanishingPointCost:
    def __init__(self, dist_thr: float):
        self.dist_thr = dist_thr

    def residuals(self, cuboid: Cuboid, T_w2cam: Pose, l2D: Tensor):
        # For details see section 5.1 in 'Non-Iterative Approach for Fast and
        # Accurate Vanishing Point Detection' by Jean-Philippe Tardif
        l3D = to_homogeneous(l2D)
        mid3D = torch.mean(l3D, dim=-2).unsqueeze(-2)
        vp3D = cuboid.R @ T_w2cam.R.transpose(-1, -2)
        line3D = torch.cross(mid3D, vp3D.unsqueeze(-3), dim=-1)

        dist_denom = torch.linalg.norm(line3D[..., :2], dim=-1)
        dist_signed = (l3D[..., 0:1, :] * line3D).sum(-1) / dist_denom
        dist = torch.abs(dist_signed)
        dist, idx = torch.min(dist, dim=-1)

        res = dist.clamp(max=self.dist_thr)
        info = (vp3D, mid3D, line3D, dist_denom, dist_signed, idx)
        return res, info

    def jacobian(
            self, l2D: Tensor, vp3D: Tensor, mid3D: Tensor, line3D: Tensor,
            dist_denom: Tensor, dist_signed: Tensor, idx: Tensor):

        l3D = to_homogeneous(l2D)

        J_vp3D_R = -skew_symmetric(vp3D.transpose(-1, -2)).transpose(-2, -3)
        J_vp3D_d = torch.zeros(J_vp3D_R.shape[:-1] + (6,)).to(J_vp3D_R)
        J_vp3D_Rd = torch.cat([J_vp3D_R, J_vp3D_d], dim=-1)

        J_line3D_vp3D = skew_symmetric(mid3D)

        dist_denom = dist_denom.unsqueeze(-1)
        J_dist_line3D = l3D[..., 0:1, :] / dist_denom
        J_dist_line3D[..., :2] -= (
            (l3D[..., 0:1, :] * line3D).sum(-1, keepdim=True)
            * line3D[..., :2]
            / dist_denom**3)
        J_dist_line3D = J_dist_line3D * dist_signed.unsqueeze(-1).sign()
        J_dist_line3D = J_dist_line3D.unsqueeze(-2)

        J_dist_Rd = J_dist_line3D @ J_line3D_vp3D @ J_vp3D_Rd.unsqueeze(-4)
        J_dist_Rd = J_dist_Rd.squeeze(-2)

        mask = dist_signed.abs() > self.dist_thr
        J_dist_Rd.masked_fill_(mask.unsqueeze(-1), 0.0)

        idx = idx.unsqueeze(-1).unsqueeze(-1).repeat((1,) * idx.ndim + (1, 9))
        J_dist_Rd = torch.gather(J_dist_Rd, -2, idx).squeeze(-2)

        return J_dist_Rd

    def residual_jacobian(self, cuboid: Cuboid, T_w2cam: Pose, l2D: Tensor):
        res, info = self.residuals(cuboid, T_w2cam, l2D)
        J = self.jacobian(l2D, *info)
        return res, J
