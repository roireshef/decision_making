from abc import abstractmethod

import numpy as np

from decision_making.src.global_constants import EXP_CLIP_TH
from decision_making.src.planning.utils.geometry_utils import CartesianFrame
from decision_making.src.state.state import DynamicObject as ObjectState


class BoxObstacle:
    def __init__(self, x: float, y: float, theta: float, length: float, width: float):
        """
        :param x: relative location in vehicle's longitudinal axis
        :param y: relative location in vehicle's lateral axis
        :param theta: object yaw
        :param length: length of the box in its own longitudinal axis (box's x)
        :param width: length of the box in its own lateral axis (box's y)
        """
        self._x = x
        self._y = y
        self._theta = theta
        self._length = length
        self._width = width
        self._R = CartesianFrame.homo_matrix_2d(self.theta, np.array([self.x, self.y]))

    @property
    def x(self): return self._x

    @property
    def y(self): return self._y

    @property
    def theta(self): return self._theta

    @property
    def length(self): return self._length

    @property
    def width(self): return self._width

    @abstractmethod
    def compute_cost(self, points):
        """
        Takes a list of points in vehicle's coordinate frame and returns cost of proximity (to self) for each point
        :param points: either [t, p, 2] shaped numpy tensor (t trajectories, p points in each, point is [x, y]), or
        [p, 2] shaped numpy matrix (single trajectory)
        :return: numpy vector of corresponding costs to the original points
        """
        pass


class SigmoidStatic2DBoxObstacle(BoxObstacle):
    """
    Static 2D obstacle represented in vehicle's coordinate frame that computes sigmoid costs for
    points in the vehicle's coordinate frame
    """

    # width is on y, height is on x
    def __init__(self, x: float, y: float, theta: float, length: float, width: float, k: float, margin: float):
        """
        :param x: relative location in vehicle's longitudinal axis
        :param y: relative location in vehicle's lateral axis
        :param theta: object yaw
        :param length: length of the box in its own longitudinal axis (box's x)
        :param width: length of the box in its own lateral axis (box's y)
        :param k: sigmoid's  exponent coefficient
        :param margin: center of sigmoid offset
        """
        super().__init__(x, y, theta, length, width)
        self._k = k
        self._margin = margin

    @property
    def k(self): return self._k

    @property
    def margin(self): return self._margin

    @classmethod
    def from_object(cls, os: ObjectState, k, offset):
        """
        Additional constructor that takes a ObjectState from the State object and wraps it
        :param os: ObjectState object from State object
        :param k:
        :param offset:
        :return: new SigmoidStatic2DBoxObstacle instance
        """
        return cls(os.x, os.y, os.yaw, os.size.length, os.size.width, k, offset)

    def compute_cost(self, points: np.ndarray) -> np.ndarray:
        """
        Takes a list of points in vehicle's coordinate frame and returns cost of proximity (to self) for each point
        :param points: either a numpy matrix of trajectory points of shape [p, 2] ( p points, [x, y] in each point),
        or a numpy tensor of trajectories of shape [t, p, 2] (t trajectories, p points, [x, y] in each point)
        :return: numpy vector of corresponding trajectory-costs
        """
        if len(points.shape) == 2:
            points = np.array([points])

        # add a third value (=1.0) to each point in each trajectory for multiplication with homo-matrix
        ones = np.ones(points.shape[:2])
        points_ext = np.dstack((points, ones))

        # (for each trajectory:) project all points to the box obstacle coordinate-frame (absolute value)
        # now each record is the [x, y] distances from the box coordinate frame (box-center).
        points_proj = np.abs(np.einsum('ijk, kl -> ijl', points_ext, np.linalg.inv(self._R).transpose())[:, :, :2])

        # subtract from the distances: 1. the box dimensions (height, width) and the margin
        points_offset = np.subtract(points_proj, [self.length / 2 + self.margin, self.width / 2 + self.margin])

        # compute a sigmoid for each dimension [x, y] of each point (in each trajectory)
        logit_costs = np.divide(1.0, (1.0 + np.exp(np.clip(self.k * points_offset, -np.inf, EXP_CLIP_TH))))

        return np.sum(logit_costs[:, :, 0] * logit_costs[:, :, 1], axis=1)


