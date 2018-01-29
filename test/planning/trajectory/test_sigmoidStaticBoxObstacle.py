import numpy as np

from decision_making.src.state.state import ObjectSize, DynamicObject
from decision_making.test.planning.trajectory.utils import RouteFixture, PlottableSigmoidStaticBoxObstacle


def test_computeCost_threeSRoutesOneObstacle_validScore():
    routes = np.array([RouteFixture.get_route(lng=200, k=.05, step=10, lat=100, offset=0.0),
                       RouteFixture.get_route(lng=200, k=.05, step=10, lat=100, offset=-50.0),
                       RouteFixture.get_route(lng=200, k=.05, step=10, lat=100, offset=-100.0)])

    pose = np.array([200, -40, np.pi / 8])
    obj = DynamicObject(None, None, pose[0], pose[1], 0, pose[2], ObjectSize(length=40, width=20, height=20),
                        1.0, 0.0, 0.0, 0, 0)
    obs = PlottableSigmoidStaticBoxObstacle(obj, k=100, margin=np.array([10, 10]))

    costs = obs.compute_cost(routes)

    assert np.round(costs[0], 10) == 0      # obstacle-free route
    assert np.round(costs[1], 10) == 6      # obstacle-colliding route (6 points)
    assert np.round(costs[2], 10) == 0      # obstacle-free route

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