from src.planning.utils.geometry_utils import CartesianFrame
import numpy as np


class SigmoidStatic2DBoxObstacle(object):
    """
    Static 2D obstacle represented in vehicle's coordinate frame that computes sigmoid costs for
    points in the vehicle's coordinate frame
    """
    # width is on y, height is on x
    def __init__(self, x, y, theta, height, width, k, offset, w):
        """
        :param x: relative location in vehicle's longitudinal axis
        :param y: relative location in vehicle's lateral axis
        :param theta: object yaw
        :param height: length of the box in its own longitudinal axis (box's x)
        :param width: length of the box in its own lateral axis (box's y)
        :param k: sigmoid's  exponent coefficient
        :param offset: center of sigmoid offset
        :param w: the cost coefficient of this specific box (will appear in the sigmoid as the numerator)
        """
        self._x = x
        self._y = y
        self._theta = theta
        self._height = height
        self._width = width
        self._k = k
        self._w = w
        self._offset = offset
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

    @property
    def k(self): return self._k

    @property
    def w(self): return self._w

    @property
    def offset(self): return self._offset

    @classmethod
    def from_object_state(cls, os):
        """
        Additional constructor that takes a ObjectState from the State object and wraps it
        :param os: ObjectState object from State object
        :return: new SigmoidStatic2DBoxObstacle instance
        """
        pass #return cls(os.x, os.y, ...)

    def compute_cost(self, points):
        """
        Takes a list of points in vehicle's coordinate frame and returns cost of proximity (to self) for each point
        :param points: numpy array where each row is a point [x, y] relative to vehicle's coordinate frame
        :return: numpy vector of corresponding costs to the original points
        """
        pass


class CostParams:
    def __init__(self, T):
        self.T = T