from typing import Optional, Tuple

import torch
import numpy as np
from torch import Tensor

from .optimization import skew_symmetric
from .utils import to_homogeneous
from .wrappers import Pose, TensorWrapper, autocast


class Cuboid(TensorWrapper):
    def __init__(self, data: Tensor):
        assert data.shape[-1] == 15
        super().__init__(data)

    @classmethod
    @autocast
    def from_Rd(cls, R: Tensor, d: Tensor):
        '''Cuboid from a rotation matrix and plane offsets.
        Accepts numpy arrays or PyTorch tensors.

        R is the world-to-cuboid rotation matrix (i.e. Xᴸ = R Xᴳ)
        d defines the six cuboid planes xᴸ = d[0]
                                        yᴸ = d[1]
                                        zᴸ = d[2]
                                        xᴸ = d[3], d[3] > d[0]
                                        yᴸ = d[4], d[4] > d[1]
                                        zᴸ = d[5], d[5] > d[2]

        Args:
            R: rotation matrix with shape (..., 3, 3).
            d: plane offsets vector with shape (..., 6).
        '''
        assert R.shape[-2:] == (3, 3)
        assert d.shape[-1] == 6
        assert R.shape[:-2] == d.shape[:-1]
        data = torch.cat([R.flatten(start_dim=-2), d], -1)
        return cls(data)

    @classmethod
    @autocast
    def from_Rts(cls, R: Tensor, t: Tensor, s: Tensor):
        '''Cuboid from rotation matrix, translation vector and size vector.
        Accepts numpy arrays or PyTorch tensors.

        R and t defines the world-to-cuboid transformation Xᴸ = R Xᴳ + t
        s is the size of the cuboid in the local frame

        Args:
            R: rotation matrix with shape (..., 3, 3).
            t: translation vector with shape (..., 3).
            s: size vector with shape (..., 3).
        '''
        d = torch.cat([-t - s / 2.0, -t + s / 2.0], dim=-1)
        return cls.from_Rd(R, d)

    @property
    def R(self) -> Tensor:
        '''Underlying rotation matrix with shape (..., 3, 3).'''
        rvec = self._data[..., :9]
        return rvec.reshape(rvec.shape[:-1] + (3, 3))

    @property
    def d(self) -> Tensor:
        '''Underlying plane offset vector with shape (..., 6).'''
        return self._data[..., -6:]

    @property
    def t(self) -> Tensor:
        '''Translation vector with shape (..., 3).'''
        d = self.d
        t = -(d[..., :3] + d[..., 3:]) / 2.0
        return t

    @property
    def s(self) -> Tensor:
        '''Size vector with shape (..., 3)'''
        d = self.d
        s = d[..., 3:] - d[..., :3]
        return s

    def __matmul__(self, transform: 'Pose') -> 'Cuboid':
        '''Apply transform.

        Xᶜ = RᵇᶜXᵇ (cuboid)
        Xᵇ = RᵃᵇXᵃ + tᵃᵇ (transform)
        => Xᶜ = RᵇᶜRᵃᵇXᵃ + Rᵇᶜtᵃᵇ
        '''
        R = self.R @ transform.R
        t = (self.R @ transform.t.unsqueeze(-1)).squeeze(-1)
        d = self.d - torch.cat([t, t], dim=-1)
        return self.__class__.from_Rd(R, d)

    def numpy(self) -> Tuple[np.ndarray]:
        return self.R.numpy(), self.d.numpy()

    @property
    def corners(self) -> Tensor:
        '''Get the corners of the cuboid in global coordinates (..., 8, 3).'''
        p3d = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        ).to(dtype=self.dtype, device=self.device)
        d = self.d.unsqueeze(-2)
        p3d = (1.0 - p3d) * d[..., :3] + p3d * d[..., 3:]
        return p3d @ self.R

    @property
    def edges(self) -> Tensor:
        '''Get the cuboid edge indices (12, 2).'''
        edges = torch.tensor(
            [
                [0, 1],
                [0, 2],
                [0, 4],
                [1, 3],
                [1, 5],
                [2, 3],
                [2, 6],
                [3, 7],
                [4, 6],
                [4, 5],
                [5, 7],
                [6, 7],
            ],
            device=self.device,
        )
        return edges

    @property
    def faces(self) -> Tensor:
        '''Get the cuboid face indices (12, 3).'''
        faces = torch.tensor(
            [
                [0, 1, 2],
                [3, 2, 1],
                [4, 6, 5],
                [7, 5, 6],
                [0, 4, 1],
                [5, 1, 4],
                [2, 3, 6],
                [7, 6, 3],
                [0, 2, 4],
                [6, 4, 2],
                [1, 5, 3],
                [7, 3, 5],
            ],
            device=self.device,
        )
        return faces

    @autocast
    def inside(self, p3d: Tensor) -> Tensor:
        '''Check if points are inside the cuboid.

        Args:
            p3d: points in world coordinates (..., 3).
        Returns:
            inside: boolean tensor indicating whether points are inside.
        '''
        p3d = p3d @ self.R.transpose(-1, -2)
        d = self.d.unsqueeze(-2)
        inside = torch.all(d[..., :3] < p3d, axis=-1) & torch.all(
            p3d < d[..., 3:], axis=-1)
        return inside

    @autocast
    def resize(self, p3d: Tensor, expand: bool, margin: float = 0.0) -> 'Cuboid':
        '''Resize cuboid so that points fit inside.

        Args:
            p3d: points in world coordinates (..., 3).
            expand: whether to only expand the cuboid.
            margin: minimum distance between points and cuboid faces.
        Returns:
            cuboid: the resized cuboid.
        '''
        R, d = self.R, self.d
        p3d = p3d @ self.R.transpose(-1, -2)

        d_lo = torch.min(p3d, dim=-2).values - margin
        d_hi = torch.max(p3d, dim=-2).values + margin

        if expand:
            d_lo = torch.minimum(d_lo, d[..., :3])
            d_hi = torch.maximum(d_hi, d[..., 3:])

        cuboid = Cuboid.from_Rd(R, torch.cat([d_lo, d_hi], dim=-1))
        return cuboid

    def sample_edges(
        self, num_points: int, return_jacobian: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:
        '''Sample points on the edges of the cuboid.

        Args:
            num_points: number of points to sample on each edge.
            return_jacobian: whether to return the jacobian.
        Returns:
            p3d_w: the sampled points in world coordinates (..., 12N, 3).
            J: the jacobian of p3d_w with respect to R and d (..., 12N, 3, 9).
        '''
        a = (
            torch.arange(num_points, device=self.device, dtype=self.dtype) + 0.5
        ) / num_points
        zeros, ones = torch.zeros_like(a), torch.ones_like(a)
        p3d = []

        p3d.append(torch.stack([a, zeros, zeros], dim=-1))
        p3d.append(torch.stack([a, zeros, ones], dim=-1))
        p3d.append(torch.stack([a, ones, zeros], dim=-1))
        p3d.append(torch.stack([a, ones, ones], dim=-1))

        p3d.append(torch.stack([zeros, a, zeros], dim=-1))
        p3d.append(torch.stack([zeros, a, ones], dim=-1))
        p3d.append(torch.stack([ones, a, zeros], dim=-1))
        p3d.append(torch.stack([ones, a, ones], dim=-1))

        p3d.append(torch.stack([zeros, zeros, a], dim=-1))
        p3d.append(torch.stack([zeros, ones, a], dim=-1))
        p3d.append(torch.stack([ones, zeros, a], dim=-1))
        p3d.append(torch.stack([ones, ones, a], dim=-1))

        p3d = torch.cat(p3d, dim=0).repeat(self.shape + (1, 1))

        d = self.d.unsqueeze(-2)
        p3d_w = ((1.0 - p3d) * d[..., :3] + p3d * d[..., 3:]) @ self.R

        if return_jacobian:
            Rt = self.R.unsqueeze(-3).transpose(-1, -2)
            Jd = Rt @ torch.cat([
                torch.diag_embed(1.0 - p3d),
                torch.diag_embed(p3d)
            ], dim=-1)
            JR = skew_symmetric(p3d_w) @ Rt
            J = torch.cat([JR, Jd], dim=-1)
        else:
            J = None

        return p3d_w, J

    def project(
        self, p2d: Tensor, T_w2cam: 'Pose', return_jacobian: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:
        '''Project image points onto the faces of the cuboid.

        It is assumed that the camera is inside the cuboid, so every 2D point
        will have a corresponding 3D point.

        Distortion is not supported, and thus there is no Camera passed to
        the function.

        Args:
            p2d: normalized image points (..., 2).
            T_w2cam: camera pose.
            return_jacobian: whether to return the jacobian.
        Returns:
            p3d_w: the projected 3D points in world coordinates (..., 3).
            J: the jacobian of p3d_w with respect to R and d (..., 3, 9).
        '''
        R_w2cam, t_w2cam = T_w2cam.R, T_w2cam.t.unsqueeze(-1)
        R, d = self.R, self.d
        eps = 1e-9

        n3d = torch.cat([torch.eye(3), torch.eye(3)], dim=0).to(p2d)
        n3d_w = (n3d @ R).unsqueeze(-1)

        p3d_cam = to_homogeneous(p2d)
        p3d_w = torch.zeros_like(p3d_cam)
        λ_min = torch.full(p2d.shape[:-1], torch.inf).to(p2d)
        if return_jacobian:
            J = torch.zeros(p2d.shape[:-1] + (3, 9)).to(p2d)
        else:
            J = None

        for i in range(6):
            # For normalized 2D points x and 3D points X we have
            #  λx = RX + t <=> X = λRᵀx - Rᵀt
            # We solve for λ such that the 3D points lie on the plane
            #  nᵀX - d = 0
            #  => λ = (nᵀRᵀt + d) / nᵀRᵀx
            n3d_wi = n3d_w[..., i, :, :]

            ntRt = n3d_wi.transpose(-2, -1) @ R_w2cam.transpose(-2, -1)
            λ = (ntRt @ t_w2cam + d[..., i, None, None]) / (
                p3d_cam @ ntRt.transpose(-2, -1) + eps
            )
            p3d_wi = (λ * p3d_cam - t_w2cam.transpose(-1, -2)) @ R_w2cam

            λ = λ.squeeze(-1)
            valid = (0 < λ) & (λ < λ_min)
            λ_min = torch.where(valid, λ, λ_min)
            p3d_w[valid] = p3d_wi[valid]

            if return_jacobian:
                J_p3d_λ = (p3d_cam @ R_w2cam).unsqueeze(-1)

                J_λ_d = 1.0 / (p3d_cam @ ntRt.transpose(-2, -1) + eps)
                Rn = R_w2cam @ n3d_wi
                J_λ_n = (
                    p3d_cam @ Rn * t_w2cam.transpose(-2, -1)
                    - (Rn.transpose(-2, -1) @ t_w2cam + d[..., i, None, None])
                    * p3d_cam
                ) / (p3d_cam @ Rn + eps) ** 2
                J_λ_nd = torch.cat([J_λ_n @ R_w2cam, J_λ_d], dim=-1).unsqueeze(
                    -2
                )

                J_n_R = skew_symmetric(n3d_wi[..., 0]) @ R.transpose(-2, -1)
                J_n_d = torch.zeros(self.shape + (3, 6)).to(p2d)
                J_n_Rd = torch.cat([J_n_R, J_n_d], dim=-1)
                J_d_R = torch.zeros(self.shape + (1, 3)).to(p2d)
                J_d_d = torch.zeros(self.shape + (1, 6)).to(p2d)
                J_d_d[..., 0, i] = 1.0
                J_d_Rd = torch.cat([J_d_R, J_d_d], dim=-1)
                J_nd_Rd = torch.cat([J_n_Rd, J_d_Rd], dim=-2).unsqueeze(-3)

                J_p3d_Rd = J_p3d_λ @ J_λ_nd @ J_nd_Rd
                J[valid] = J_p3d_Rd[valid]

        return p3d_w, J
