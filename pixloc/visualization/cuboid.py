import cv2
import numpy as np

from ..pixlib.geometry import Camera
from ..pixlib.geometry.cuboid import Cuboid


def draw_line(l3d: np.ndarray, camera: Camera, img: cv2.typing.MatLike,
              color: cv2.typing.Scalar, thickness: int) -> None:
    """Draw a 3D line segment.
    Args:
        l3d: the line segment, given in the camera frame, with shape (2, 3).
        camera: the target camera.
        img: the target image.
        color: line color.
        thickness: line thickness in pixels.
    """
    eps = 1e-2
    if l3d[0, 2] < eps and l3d[1, 2] < eps:
        return
    if l3d[0, 2] < eps or l3d[1, 2] < eps:
        t = (eps - l3d[0, 2]) / (l3d[1, 2] - l3d[0, 2])
        l3d[0 if l3d[0, 2] < eps else 1] = l3d[0] + t * (l3d[1] - l3d[0])
    l2d, _ = camera.world2image(l3d)
    l2d = np.rint(l2d.numpy()).astype(np.int32)
    cv2.line(img, l2d[0], l2d[1], color, thickness)


def draw_edges(cuboid: Cuboid, camera: Camera, img: cv2.typing.MatLike,
               color: cv2.typing.Scalar, thickness: int) -> None:
    """Draw the edges of a cuboid.
    Args:
        cuboid: the cuboid, given in the camera frame.
        camera: the target camera.
        img: the target image.
        color: edge color.
        thickness: line thickness in pixels.
    """
    for line in cuboid.corners[cuboid.edges].numpy():
        draw_line(line, camera, img, color, thickness)
