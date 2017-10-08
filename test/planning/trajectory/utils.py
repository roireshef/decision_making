from typing import List

import matplotlib.patches as patches
import numpy as np

from decision_making.src.planning.trajectory.cost_function import SigmoidStatic2DBoxObstacle
from decision_making.src.planning.utils.columns import R_THETA


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


class PlottableSigmoidStatic2DBoxObstacle(SigmoidStatic2DBoxObstacle):
    def plot(self, plt):
        plt.plot(self.poses[0][0], self.poses[0][1], '*k')
        plt.plot(self.poses[-1][0], self.poses[-1][1], '*r')
        lower_left_p = np.dot(self._R[0], [-self.length / 2, -self.width / 2, 1])
        plt.add_patch(patches.Rectangle(
            (lower_left_p[0], lower_left_p[1]), self.length, self.width, angle=np.rad2deg(self.poses[0][R_THETA]), hatch='\\',
            fill=False
        ))

        lower_left_p = np.dot(self._R[0], [-self.length / 2 - self._margin, -self.width / 2 - self._margin, 1])
        plt.add_patch(patches.Rectangle(
            (lower_left_p[0], lower_left_p[1]), self.length + 2 * self._margin, self.width + 2 * self._margin,
            angle=np.rad2deg(self.poses[0][R_THETA]), fill=True, alpha=0.15, color=[0, 0, 0]
        ))


class WerlingVisualizer:
    @staticmethod
    def plot_route(plt, route):
        plt.plot(route[:, 0], route[:, 1], '-k')

    @staticmethod
    def plot_obstacles(plt, obs: List[PlottableSigmoidStatic2DBoxObstacle]):
        for o in obs: o.plot(plt)

    @staticmethod
    def plot_alternatives(plt, alternatives: np.ndarray):
        for alt in alternatives:
            plt.plot(alt[:, 0], alt[:, 1], '-m')
            plt.plot(alt[-1, 0], alt[-1, 1], 'om')

    @staticmethod
    def plot_best(plt, traj):
        plt.plot(traj[:, 0], traj[:, 1], '-g')
        plt.plot(traj[-1, 0], traj[-1, 1], 'og')
