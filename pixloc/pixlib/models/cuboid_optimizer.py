import logging
from typing import Dict, Optional

import torch
from torch import Tensor

from .base_optimizer import BaseOptimizer
from .learned_optimizer import DampingNet
from ..geometry import Camera, Pose
from ..geometry.cuboid import Cuboid
from ..geometry.cuboid_costs import FeaturemetricCost, EdgeCost, VanishingPointCost
from ..geometry.optimization import optimizer_step, so3exp_map
from ..geometry import losses  # noqa

logger = logging.getLogger(__name__)


class CuboidOptimizer(BaseOptimizer):
    default_conf = dict(
        damping=dict(
            type='constant',
            log_range=[-6, 5],
        ),
        dd_stop_criteria=1e-4,
        feat_cost_scale=1.0,
        edge_cost_scale=0.0,
        edge_loss_fn='squared_loss',
        num_edge_points=40,
        vp_cost_scale=0.0,
        vp_loss_fn='squared_loss',
        vp_dist_thr=0.05,
        clamp_dd=0.0,
        expand_cuboid=True,
        expand_cuboid_margin=0.1,
        feature_dim=None,
    )

    def _init(self, conf):
        super()._init(conf)
        self.dampingnet = DampingNet(conf.damping, num_params=9)
        self.cost_fn = FeaturemetricCost(
            self.interpolator, normalize=conf.normalize_features)
        # self.loss_fn set in BaseOptimizer
        self.edge_cost_fn = EdgeCost(
            self.interpolator, self.conf.num_edge_points)
        self.edge_loss_fn = eval('losses.' + conf.edge_loss_fn)
        self.vp_cost_fn = VanishingPointCost(self.conf.vp_dist_thr)
        self.vp_loss_fn = eval('losses.' + conf.vp_loss_fn)
        assert not conf.jacobi_scaling

    def build_system_feat(self, cuboid: Cuboid, T_w2cam: Pose, camera: Camera,
                          p2D: Tensor, F: Tensor, F_interp: Tensor,
                          mask: Optional[Tensor], W: Optional[Tensor],
                          W_interp: Optional[Tensor]):

        res, valid, w_unc, J = self.cost_fn.residual_jacobian(
            cuboid, T_w2cam, camera, p2D, F, F_interp, W, W_interp)
        if mask is not None:
            valid &= mask.flatten(-2, -1).unsqueeze(-2)

        res = res.flatten(1, 2)
        valid = valid.flatten(1, 2)
        J = J.flatten(1, 2)

        # compute the cost and aggregate the weights
        cost = (res**2).sum(-1)
        cost, w_loss, _ = self.loss_fn(cost)
        weights = self.conf.feat_cost_scale * w_loss * valid.float()
        if w_unc is not None:
            weights *= w_unc.flatten(1, 2)

        # solve the linear system
        g, H = self.build_system(J, res, weights)
        return g, H, valid

    def build_system_edge(self, cuboid: Cuboid, T_w2cam: Pose, camera: Camera,
                          E: Tensor, W: Tensor):

        res, valid, w_unc, J = self.edge_cost_fn.residual_jacobian(
            cuboid, T_w2cam, camera, E, W)

        res = res.flatten(1, 2)
        valid = valid.flatten(1, 2)
        J = J.flatten(1, 2)

        cost = (res**2).sum(-1)
        cost, w_loss, _ = self.edge_loss_fn(cost)
        weights = self.conf.edge_cost_scale * w_loss * valid.float()
        if w_unc is not None:
            weights *= w_unc.flatten(1, 2)

        return self.build_system(J, res, weights)

    def build_system_vp(self, cuboid: Cuboid, T_w2cam: Pose, l2D: Tensor,
                        l2D_mask: Tensor):

        res, J = self.vp_cost_fn.residual_jacobian(
            cuboid.unsqueeze(-1), T_w2cam, l2D)

        res = res.flatten(1, 2)
        valid = l2D_mask.flatten(1, 2)
        J = J.flatten(1, 2)

        cost = res**2
        cost, w_loss, _ = self.vp_loss_fn(cost)
        weights = self.conf.vp_cost_scale * w_loss * valid.float()

        return self.build_system(J.unsqueeze(-2), res.unsqueeze(-1), weights)

    def step(self, g: Tensor, H: Tensor, lambda_: Tensor,
             mask: Optional[Tensor] = None):

        delta = optimizer_step(g, H, lambda_, mask=mask)
        dw, dd = delta.split([3, 6], dim=-1)
        if self.conf.clamp_dd:
            # dd = dd.clamp(min=-self.conf.clamp_dd, max=self.conf.clamp_dd)
            dd = torch.tanh(2.0 / self.conf.clamp_dd * dd) * self.conf.clamp_dd
        dR = so3exp_map(dw)
        return dR, dd

    def early_stop(self, **args):
        stop = False
        if not self.training and (args['i'] % 10) == 0:
            dR, dd, grad = args['dR'], args['dd'], args['grad']
            grad_norm = torch.norm(grad.detach(), dim=-1)
            small_grad = grad_norm < self.conf.grad_stop_criteria
            trace = torch.diagonal(dR, dim1=-1, dim2=-2).sum(-1)
            cos = torch.clamp((trace - 1) / 2, -1, 1)
            dr = torch.rad2deg(torch.acos(cos).abs())
            small_step = ((dr < self.conf.dR_stop_criteria)
                          & torch.all(dd.abs() < self.conf.dd_stop_criteria, dim=-1))
            if torch.all(small_step | small_grad):
                stop = True
        return stop

    def _forward(self, data: Dict):
        return self._run(
            data['cuboid_init'], data['T_w2cam'], data['camera'],
            data['p2D'], data['F'], data['F_interp'], data['mask'],
            data['Wf'], data['Wf_interp'], data['E'], data['We'],
            data['l2D'], data['l2D_mask'])

    def _run(self, cuboid_init: Cuboid, T_w2cam: Pose, camera: Camera,
             p2D: Tensor, F: Tensor, F_interp: Tensor, mask: Optional[Tensor],
             Wf: Optional[Tensor], Wf_interp: Optional[Tensor],
             E: Tensor, We: Optional[Tensor], l2D: Tensor, l2D_mask: Tensor):

        cuboid = cuboid_init
        failed = torch.full(
            cuboid.shape, False, dtype=torch.bool, device=cuboid.device)
        lambda_ = self.dampingnet()

        cams_w = (-T_w2cam.t.unsqueeze(-2) @ T_w2cam.R).squeeze(-2)

        for i in range(self.conf.num_iters):
            g = torch.zeros(cuboid.shape + (9,)).to(p2D)
            H = torch.zeros(cuboid.shape + (9, 9)).to(p2D)

            if self.conf.feat_cost_scale > 0:
                g_feat, H_feat, valid = self.build_system_feat(
                    cuboid, T_w2cam, camera, p2D, F, F_interp, mask, Wf, Wf_interp)
                g += g_feat
                H += H_feat
                failed = failed | (valid.long().sum(-1) < 10)  # too few points

            if self.conf.edge_cost_scale > 0:
                g_edge, H_edge = self.build_system_edge(
                    cuboid, T_w2cam, camera, E, We)
                g += g_edge
                H += H_edge

            if self.conf.vp_cost_scale > 0:
                g_vp, H_vp = self.build_system_vp(
                    cuboid, T_w2cam, l2D, l2D_mask)
                g += g_vp
                H += H_vp

            dR, dd = self.step(g, H, lambda_, mask=~failed)
            cuboid = Cuboid.from_Rd(dR @ cuboid.R, cuboid.d + dd)

            if not self.training and self.conf.expand_cuboid:
                cuboid = cuboid.resize(
                    cams_w, True, self.conf.expand_cuboid_margin)

            self.log(i=i, cuboid_init=cuboid_init, cuboid=cuboid, dR=dR, dd=dd)
            if self.early_stop(i=i, dR=dR, dd=dd, grad=g):
                break

        if failed.any():
            logger.debug('One batch element had too few valid points.')

        return cuboid, failed
