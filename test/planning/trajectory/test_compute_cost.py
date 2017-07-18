import unittest

from src.planning.trajectory.cost_function import SigmoidStatic2DBoxObstacle
from test.planning.route_fixture import RouteFixture

import numpy as np


class TestCostComputation(unittest.TestCase):
    def test(self):
        routes = np.array([RouteFixture.get_route(lng=200, k=.05, step=10, lat=100, offset=0.0),
                           RouteFixture.get_route(lng=200, k=.05, step=10, lat=100, offset=-50.0),
                           RouteFixture.get_route(lng=200, k=.05, step=10, lat=100, offset=-100.0)])

        obs = PlottableSigmoidStatic2DBoxObstacle(x=200, y=-40, theta=np.pi / 8, height=40, width=20, k=100, margin=10)

        costs = obs.compute_cost(routes)

        self.assertEqual(np.round(costs[0], 10), 0)     # obstacle-free route
        self.assertEqual(np.round(costs[1], 10), 6)     # obstacle-colliding route (6 points)
        self.assertEqual(np.round(costs[2], 10), 0)     # obstacle-free route

        # import matplotlib.pyplot as plt
        #
        # fig = plt.figure()
        # p1 = fig.add_subplot(111)
        # for route_points in routes:
        #     p1.plot(route_points[:, 0], route_points[:, 1], '-*k')
        #
        # obs.plot(p1)
        #
        # fig.show()
        # fig.clear()


class PlottableSigmoidStatic2DBoxObstacle(SigmoidStatic2DBoxObstacle):

    def plot(self, plt):
        import matplotlib.patches as patches

        plt.plot(self.x, self.y, '*k')
        lower_left_p = np.dot(self._R, [-self.height / 2, -self.width / 2, 1])
        plt.add_patch(patches.Rectangle(
            (lower_left_p[0], lower_left_p[1]), self.height, self.width, angle=np.rad2deg(self.theta), hatch='\\',
            fill=False
        ))

        lower_left_p = np.dot(self._R, [-self.height / 2 - self._margin, -self.width / 2 - self._margin, 1])
        plt.add_patch(patches.Rectangle(
            (lower_left_p[0], lower_left_p[1]), self.height + 2 * self._margin, self.width + 2 * self._margin,
            angle=np.rad2deg(self.theta), fill=True, alpha=0.15, color=[0, 0, 0]
        ))