from src.planning.utils.geometry_utils import *
import numpy as np
import unittest


class TestGeometryUtils(unittest.TestCase):

    ACCURACY_TH = 0.1

    def test(self):
        self.assertFrenetAccuracy()

    def assertFrenetAccuracy(self):
        route_points = self._get_route()
        cpoints = np.array([[220.0, 0.0], [150.0, 0.0],
                            [280.0, 40.0], [320.0, 60.0],
                            [350.0, 0.0]
                            ])

        frenet = FrenetMovingFrame(route_points)

        fpoints = np.array([frenet.cpoint_to_fpoint(cpoint) for cpoint in cpoints])
        new_cpoints = np.array([frenet.fpoint_to_cpoint(fpoint) for fpoint in fpoints])

        errors = np.linalg.norm(cpoints-new_cpoints, axis=1)

        for error in errors:
            self.assertLess(error, self.ACCURACY_TH, 'FrenetMovingFrame point conversions aren\'t accurate enough')

        # import matplotlib.pyplot as plt
        #
        # fig = plt.figure()
        # p1 = fig.add_subplot(111)
        # p1.plot(route_points[:, 0], route_points[:, 1], '-r')
        # p1.plot(frenet.curve[:, 0], frenet.curve[:, 1], '-k')
        #
        # for i in range(len(cpoints)):
        #     p1.plot(cpoints[i, 0], cpoints[i, 1], '*b')
        #     p1.plot(new_cpoints[i, 0], new_cpoints[i, 1], '.r')
        #
        # print(errors)
        #
        # fig.show()
        # fig.clear()

    @staticmethod
    def _get_route(lng=200, k=0.05, step=40, lat=100, offset=-50.0):
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