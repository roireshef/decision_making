from typing import List

import matplotlib.patches as patches
import numpy as np

from decision_making.src.planning.types import CURVE_YAW, CartesianPoint2D, CartesianExtendedState, C_X, C_Y
from decision_making.src.planning.trajectory.cost_function import SigmoidDynamicBoxObstacle, SigmoidStaticBoxObstacle, \
    SigmoidBoxObstacle
from decision_making.src.prediction.predictor import Predictor
from decision_making.src.state.state import DynamicObject


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


class PlottableSigmoidBoxObstacle(SigmoidBoxObstacle):
    def plot(self, plt):
        pass


class PlottableSigmoidStaticBoxObstacle(SigmoidStaticBoxObstacle, PlottableSigmoidBoxObstacle):
    def __init__(self, obj: DynamicObject, k: float, margin: CartesianPoint2D):
        pose = np.array([obj.x, obj.y, obj.yaw, 0])
        super().__init__(pose, obj.size.length, obj.size.width, k, margin)
        self.pose = pose

    def plot(self, plt):
        plt.plot(self.pose[0], self.pose[1], '*k')
        H_inv = np.linalg.inv(self._H_inv.transpose())
        lower_left_p = np.dot(H_inv, [-self.length / 2, -self.width / 2, 1])
        plt.add_patch(patches.Rectangle(
            (lower_left_p[0], lower_left_p[1]), self.length, self.width, angle=np.rad2deg(self.pose[CURVE_YAW]),
            hatch='\\', fill=False
        ))

        lower_left_p = np.dot(H_inv, [-self.length / 2 - self._margin[0], -self.width / 2 - self._margin[1], 1])
        plt.add_patch(patches.Rectangle(
            (lower_left_p[0], lower_left_p[1]), self.length + 2 * self._margin[0], self.width + 2 * self._margin[1],
            angle=np.rad2deg(self.pose[CURVE_YAW]), fill=True, alpha=0.15, color=[0, 0, 0]
        ))


class PlottableSigmoidDynamicBoxObstacle(SigmoidDynamicBoxObstacle, PlottableSigmoidBoxObstacle):
    def __init__(self, obj: DynamicObject, k: float, margin: CartesianPoint2D,
                 time_samples: np.ndarray, predictor: Predictor):
        # get predictions of the dynamic object in global coordinates
        poses = predictor.predict_object(obj, time_samples)
        poses[0][CURVE_YAW] = obj.yaw
        super().__init__(poses, obj.size.length, obj.size.width, k, margin)
        self.poses = poses

    def plot(self, plt):
        plt.plot(self.poses[0][0], self.poses[0][1], '*k')
        plt.plot(self.poses[-1][0], self.poses[-1][1], '*r')
        H_inv = np.linalg.inv(self._H_inv[0].transpose())
        lower_left_p = np.dot(H_inv, [-self.length / 2, -self.width / 2, 1])
        plt.add_patch(patches.Rectangle(
            (lower_left_p[0], lower_left_p[1]), self.length, self.width, angle=np.rad2deg(self.poses[0][CURVE_YAW]),
            hatch='\\', fill=False
        ))

        lower_left_p = np.dot(H_inv, [-self.length / 2 - self._margin[0], -self.width / 2 - self._margin[1], 1])
        plt.add_patch(patches.Rectangle(
            (lower_left_p[0], lower_left_p[1]), self.length + 2 * self._margin[0], self.width + 2 * self._margin[1],
            angle=np.rad2deg(self.poses[0][CURVE_YAW]), fill=False, alpha=0.15, color=[0, 0, 0]
        ))


class WerlingVisualizer:
    @staticmethod
    def plot_route(plt, route):
        plt.plot(route[:, 0], route[:, 1], '-b')

    @staticmethod
    def plot_obstacles(plt, obs: List[PlottableSigmoidBoxObstacle]):
        for o in obs: o.plot(plt)

    @staticmethod
    def plot_alternatives(plt, alternatives: np.ndarray, costs: np.ndarray):
        max_cost = np.log(1+max(costs))
        min_cost = np.log(1+min(costs))
        for i, alt in enumerate(alternatives):
            cost = np.log(1+costs[i])
            c = 1 - (cost-min_cost)/(max_cost-min_cost)
            plt.plot(alt[:, 0], alt[:, 1], '-', color=[c, 0, 0.5])
            plt.plot(alt[-1, 0], alt[-1, 1], 'o', color=[c, 0, 0.5])

    @staticmethod
    def plot_best(plt, traj):
        plt.plot(traj[:, 0], traj[:, 1], '-g')
        plt.plot(traj[-1, 0], traj[-1, 1], 'og')

    @staticmethod
    def plot_goal(plt, goal: CartesianExtendedState):
        plt.plot(goal[C_X], goal[C_Y], 'or')
