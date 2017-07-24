import unittest

import numpy as np

from test.planning.trajectory.utils import RouteFixture, PlottableSigmoidStatic2DBoxObstacle


class TestCostComputation(unittest.TestCase):
    def test(self):
        self.test_single_box()

    def test_single_box(self):
        routes = np.array([RouteFixture.get_route(lng=200, k=.05, step=10, lat=100, offset=0.0),
                           RouteFixture.get_route(lng=200, k=.05, step=10, lat=100, offset=-50.0),
                           RouteFixture.get_route(lng=200, k=.05, step=10, lat=100, offset=-100.0)])

        obs = PlottableSigmoidStatic2DBoxObstacle(x=200, y=-40, theta=np.pi / 8, height=40, width=20, k=100, margin=10)

        costs = obs.compute_cost(routes)

        self.assertEqual(np.round(costs[0], 10), 0)  # obstacle-free route
        self.assertEqual(np.round(costs[1], 10), 6)  # obstacle-colliding route (6 points)
        self.assertEqual(np.round(costs[2], 10), 0)  # obstacle-free route

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
