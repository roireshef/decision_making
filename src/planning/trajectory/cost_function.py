from abc import abstractmethod

import numpy as np

from src.planning.utils.geometry_utils import CartesianFrame


class BoxObstcle(object):
    def __init__(self, x, y, theta, height, width):
        """
        :param x: relative location in vehicle's longitudinal axis
        :param y: relative location in vehicle's lateral axis
        :param theta: object yaw
        :param height: length of the box in its own longitudinal axis (box's x)
        :param width: length of the box in its own lateral axis (box's y)
        """
        self._x = x
        self._y = y
        self._theta = theta
        self._height = height
        self._width = width
        self._R = CartesianFrame.homo_matrix_2d(self.theta, np.array([self.x, self.y]))

    @property
    def x(self): return self._x

    @property
    def y(self): return self._y

    @property
    def theta(self): return self._theta

    @property
    def height(self): return self._height

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


class SigmoidStatic2DBoxObstacle(BoxObstcle):
    """
    Static 2D obstacle represented in vehicle's coordinate frame that computes sigmoid costs for
    points in the vehicle's coordinate frame
    """

    # width is on y, height is on x
    def __init__(self, x, y, theta, height, width, k, margin):
        """
        :param x: relative location in vehicle's longitudinal axis
        :param y: relative location in vehicle's lateral axis
        :param theta: object yaw
        :param height: length of the box in its own longitudinal axis (box's x)
        :param width: length of the box in its own lateral axis (box's y)
        :param k: sigmoid's  exponent coefficient
        :param margin: center of sigmoid offset
        """
        super().__init__(x, y, theta, height, width)
        self._k = k
        self._margin = margin

    @property
    def k(self): return self._k

    @property
    def margin(self): return self._margin

    @classmethod
    def from_object_state(cls, os, k, offset):
        """
        Additional constructor that takes a ObjectState from the State object and wraps it
        :param os: ObjectState object from State object
        :param k:
        :param offset:
        :return: new SigmoidStatic2DBoxObstacle instance
        """
        pass  # return cls(os.x, os.y, ...)

    def compute_cost(self, points):
        """
        Takes a list of points in vehicle's coordinate frame and returns cost of proximity (to self) for each point
        :param points: numpy array where each row is a point [x, y] relative to vehicle's coordinate frame
        :return: numpy vector of corresponding costs to the original points
        """
        if len(points.shape) == 2:
            points = np.array([points])

        # add a third value (=1.0) to each point in each trajectory
        ones = np.ones(points.shape[:2])
        points_ext = np.dstack((points, ones))

        # (for each trajectory:) project all points to the box obstacle coordinate-frame (absolute value)
        # now each record is the [x, y] distances from the box coordinate frame (box-center).
        points_proj = np.abs(np.einsum('ijk, kl -> ijl', points_ext, np.linalg.inv(self._R).transpose())[:, :, :2])

        # subtract from the distances: 1. the box dimensions (height, width) and the margin
        points_offset = np.subtract(points_proj, [self.height / 2 + self.margin, self.width / 2 + self.margin])

        # compute a sigmoid for each dimension [x, y] of each point (in each trajectory)
        logit_costs = np.divide(1.0, (1.0 + np.exp(self.k * points_offset)))

        # the multiplication of [x, y] costs for each point is the cost we want
        return np.einsum('ij, ik -> i', logit_costs[:, :, 0], logit_costs[:, :, 1])


class CostParams:
    def __init__(self, T: float, ref_deviation_weight: float, lane_deviation_weight: float, obstacle_weight: float,
                 left_lane_offset: float, right_lane_offset: float, left_deviation_coef: float,
                 right_deviation_coeft: float,
                 obstacle_offset: float, obstacle_exp: float):
        self._T = T
        self._ref_deviation_weight = ref_deviation_weight
        self._lane_deviation_weight = lane_deviation_weight
        self._obstacle_weight = obstacle_weight
        self._left_lane_offset = left_lane_offset
        self._right_lane_offset = right_lane_offset
        self._left_deviation_exp = left_deviation_coef
        self._right_deviation_exp = right_deviation_coeft
        self._obstacle_offset = obstacle_offset
        self._obstacle_exp = obstacle_exp

    @property
    def T(self): return self._T

    @property
    def ref_deviation_weight(self): return self._ref_deviation_weight

    @property
    def lane_deviation_weight(self): return self._lane_deviation_weight

    @property
    def obstacle_weight(self): return self._obstacle_weight

    @property
    def left_lane_offset(self): return self._left_lane_offset

    @property
    def right_lane_offset(self): return self._right_lane_offset

    @property
    def left_deviation_exp(self): return self._left_deviation_exp

    @property
    def right_deviation_exp(self): return self._right_deviation_exp
