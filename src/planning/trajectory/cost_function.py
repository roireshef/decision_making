from src.planning.utils.geometry_utils import CartesianFrame
import numpy as np


class SigmoidStatic2DBoxObstacle:
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
        self.x = x
        self.y = y
        self.theta = theta
        self.height = height
        self.width = width
        self.k = k
        self.w = w
        self.offset = offset
        self.R = CartesianFrame.homo_matrix_2d(self.theta, np.array([self.x, self.y]))

    @staticmethod
    def parse(object_state):
        """
        Additional constructor that takes a ObjectState from the State object and wraps it
        :param object_state: ObjectState object from State object
        :return:
        """
        pass

    def compute_cost(self, points):
        """
        Takes a list of points in vehicle's coordinate frame and returns cost of proximity (to self) for each point
        :param points: numpy array where each row is a point [x, y] relative to vehicle's coordinate frame
        :return: numpy vector of corresponding costs to the original points
        """
        pass