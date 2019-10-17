from typing import List

import matplotlib.patches as patches
import numpy as np

from decision_making.src.planning.types import CURVE_YAW, CartesianPoint2D, CartesianExtendedState, C_X, C_Y, C_YAW
from decision_making.src.prediction.ego_aware_prediction.ego_aware_predictor import EgoAwarePredictor
from decision_making.src.state.state import DynamicObject, State
from decision_making.src.utils.geometry_utils import CartesianFrame


class RouteFixture:
    @staticmethod
    def get_route(lng: int = 200, k: float = 0.05, step: int = 40, lat: int = 100, offset: float = -50.0):
        def stretch(v):
            min = np.min(v[:, 1])
            max = np.max(v[:, 1])

            for i in range(len(v)):
                v[i, 1] = lat * (v[i, 1] - min) / (max - min) + offset

            return v

        return np.concatenate((
            np.array([[i, offset] for i in range(0, lng, step)]),
            stretch(np.array([[i + lng, 1 / (1 + np.exp(-k * (i - lng / 2)))] for i in range(0, lng, step)])),
            np.array([[i + 2 * lng, lat + offset] for i in range(0, lng, step)])
        ), axis=0)

    @staticmethod
    def create_sigmoid_route(lng: int = 200, k: float = 10, step: float = 1, lat: int = 100, offset: float = -50.0):
        """
        create route of sigmoid shape
        :param lng: length
        :param k: k of the sigmoid
        :param step: step between points
        :param lat: initial latitude
        :param offset: ofset of the sigmoid
        :return: None
        """
        def stretch(v):
            min = np.min(v[:, 1])
            max = np.max(v[:, 1])

            for i in range(len(v)):
                v[i, 1] = lat * (v[i, 1] - min) / (max - min) + offset

            return v

        stretch(np.array([[i, 1 / (1 + np.exp(-k * (i/lng - 0.5)))] for i in np.arange(0, lng, step)]))

    @staticmethod
    def create_round_route(lng: int = 100, lat: float = 0, ext: float = 10, step: float = 1, curvature: float = 0.01):
        """
        create route of circle shape segment
        :param lng: length
        :param lat: initial latitude
        :param ext: extension in both sides
        :param step: step between points
        :param curvature: 1/radius
        :return: None
        """
        ang = (lng - 2*ext) * curvature
        r = 1/curvature
        round_part = np.array([[r*np.sin(a), lat + r - r*np.cos(a)] for a in np.arange(0, ang, step*curvature)])
        last_round = round_part[-1]
        return np.concatenate((
            np.array([[i, lat] for i in np.arange(-ext, 0, step)]),
            round_part,
            np.array([last_round + i * np.array([np.cos(ang), np.sin(ang)]) for i in np.arange(1, ext, step)])
        ), axis=0)

    @staticmethod
    def create_cubic_route(lng: float = 100, lat: float = 0, ext: float = 10, step: float = 1, curvature: float = 0.1):
        return np.array([[x, lat + curvature*lng*((x/lng)**3)] for x in np.arange(-ext, lng+ext, step)])


class PlottableSigmoidBoxObstacle:
    def __init__(self, state: State, obj: DynamicObject, k: float, margin: CartesianPoint2D,
                 time_samples: np.ndarray, predictor: EgoAwarePredictor):
        # get predictions of the dynamic object in global coordinates
        predicted_objects = predictor.predict_objects(state, [obj.obj_id], time_samples, None)[obj.obj_id]
        poses = [obj.cartesian_state for obj in predicted_objects]
        poses[0][CURVE_YAW] = obj.yaw
        self.poses = poses
        self.length = obj.size.length
        self.width = obj.size.width
        self.k = k
        self.margin = margin
        self.H_inv = np.zeros((len(poses), 3, 3))
        for pose_ind in range(len(poses)):
            H = CartesianFrame.homo_matrix_2d(poses[pose_ind][C_YAW], poses[pose_ind][:C_YAW])
            self.H_inv[pose_ind] = np.linalg.inv(H).transpose()

    def plot(self, plt):
        plt.plot(self.poses[0][0], self.poses[0][1], '*k')
        plt.plot(self.poses[-1][0], self.poses[-1][1], '*r')
        H_inv = np.linalg.inv(self.H_inv[0].transpose())
        lower_left_p = np.dot(H_inv, [-self.length / 2, -self.width / 2, 1])
        plt.add_patch(patches.Rectangle(
            (lower_left_p[0], lower_left_p[1]), self.length, self.width, angle=np.rad2deg(self.poses[0][CURVE_YAW]),
            hatch='\\', fill=False
        ))

        lower_left_p = np.dot(H_inv, [-self.length / 2 - self.margin[C_X], -self.width / 2 - self.margin[C_Y], 1])
        plt.add_patch(patches.Rectangle(
            (lower_left_p[0], lower_left_p[1]), self.length + 2 * self.margin[C_X], self.width + 2 * self.margin[C_Y],
            angle=np.rad2deg(self.poses[0][CURVE_YAW]), fill=False, alpha=0.15, color=[0, 0, 0]
        ))


class WerlingVisualizer:
    @staticmethod
    def plot_route(plt, route, param='-b'):
        plt.plot(route[:, 0], route[:, 1], param)

    @staticmethod
    def plot_obstacles(plt, obs: List[PlottableSigmoidBoxObstacle]):
        for o in obs: o.plot(plt)

    @staticmethod
    def plot_alternatives(plt, alternatives: np.ndarray, costs: np.ndarray):
        if costs is None:
            costs = np.array([0,]*alternatives.shape[0])
        max_cost = np.log(1+max(costs))
        min_cost = np.log(1+min(costs))
        for i in range(alternatives.shape[0]-1, -1, -1):
            alt = alternatives[i]
            cost = np.log(1+costs[i])
            c = 1 - (cost-min_cost)/(max_cost-min_cost)
            plt.plot(alt[:, 0], alt[:, 1], '-', color=[c, 0, 0.5])
            plt.plot(alt[-1, 0], alt[-1, 1], 'o', color=[c, 0, 0.5])

    @staticmethod
    def plot_best(plt, traj):
        plt.plot(traj[:, 0], traj[:, 1], '-g', linewidth=2)
        plt.plot(traj[-1, 0], traj[-1, 1], 'og')

    @staticmethod
    def plot_goal(plt, goal: CartesianExtendedState):
        plt.plot(goal[C_X], goal[C_Y], 'or')
