import numpy as np
import time

from decision_making.src.messages.trajectory_parameters import TrajectoryCostParams, SigmoidFunctionParams
from decision_making.src.planning.types import CURVE_X, CURVE_Y, CURVE_THETA
from decision_making.src.planning.trajectory.optimal_control.werling_planner import WerlingPlanner
from decision_making.src.prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.state.state import State, ObjectSize, EgoState, DynamicObject
from decision_making.test.constants import MAP_SERVICE_ABSOLUTE_PATH
from decision_making.test.planning.trajectory.utils import RouteFixture, PlottableSigmoidDynamicBoxObstacle, \
    WerlingVisualizer
from mapping.src.transformations.geometry_utils import CartesianFrame
from mapping.test.model.testable_map_fixtures import map_api_mock
from rte.python.logger.AV_logger import AV_Logger

from unittest.mock import patch


@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
def test_werlingPlanner_toyScenario_noException():
    logger = AV_Logger.get_logger('test_werlingPlanner_toyScenario_noException')
    route_points = CartesianFrame.add_yaw_and_derivatives(
        RouteFixture.get_route(lng=10, k=1, step=1, lat=3, offset=-.5))

    v0 = 6
    vT = 10
    v_min = 0
    v_max = 10
    a_min = -5
    a_max = 5
    T = 1.5

    predictor = RoadFollowingPredictor(logger)

    goal = np.concatenate((route_points[len(route_points) // 2, [CURVE_X, CURVE_Y, CURVE_THETA]], [vT]))

    pos1 = np.array([7, -.5])
    yaw1 = 0
    pos2 = np.array([11, 1.5])
    yaw2 = np.pi / 4

    obs = list([
        DynamicObject(obj_id=0, timestamp=0, x=pos1[0], y=pos1[1], z=0, yaw=yaw1, size=ObjectSize(1.5, 0.5, 0),
                      confidence=1.0, v_x=2.2, v_y=0, acceleration_lon=0.0, omega_yaw=0.0),
        DynamicObject(obj_id=0, timestamp=0, x=pos2[0], y=pos2[1], z=0, yaw=yaw2, size=ObjectSize(1.5, 0.5, 0),
                      confidence=1.0, v_x=1.1, v_y=0, acceleration_lon=0.0, omega_yaw=0.0)
    ])

    ego = EgoState(obj_id=-1, timestamp=0, x=0, y=0, z=0, yaw=0, size=None,
                   confidence=1.0, v_x=v0, v_y=0, steering_angle=0.0, acceleration_lon=0.0, omega_yaw=0.0)

    state = State(occupancy_state=None, dynamic_objects=obs, ego_state=ego)

    cost_params = TrajectoryCostParams(left_lane_cost=SigmoidFunctionParams(10, 1.0, 1.0),
                                       right_lane_cost=SigmoidFunctionParams(10, 1.0, 1.0),
                                       left_road_cost=SigmoidFunctionParams(10, 1.0, 1.5),
                                       right_road_cost=SigmoidFunctionParams(10, 1.0, 1.5),
                                       left_shoulder_cost=SigmoidFunctionParams(10, 1.0, 2),
                                       right_shoulder_cost=SigmoidFunctionParams(10, 1.0, 2),
                                       obstacle_cost=SigmoidFunctionParams(100, 10.0, 0.3),
                                       dist_from_ref_sq_cost=1.0,
                                       dist_from_goal_lat_sq_cost=1.0,
                                       dist_from_goal_lon_sq_cost=1.0,
                                       velocity_limits=np.array([v_min, v_max]),
                                       acceleration_limits=np.array([a_min, a_max]))

    planner = WerlingPlanner(logger, predictor)

    start_time = time.time()

    _, _, debug = planner.plan(state=state, reference_route=route_points[:, :2], goal=goal,
                               goal_time=T, cost_params=cost_params)

    end_time = time.time() - start_time

    assert True

    import matplotlib.pyplot as plt

    fig = plt.figure()
    p1 = fig.add_subplot(211)
    p2 = fig.add_subplot(212)
    time_samples = np.arange(0.0, T, 0.1)
    plottable_obs = [PlottableSigmoidDynamicBoxObstacle(o, cost_params.obstacle_cost.k,
                                                        cost_params.obstacle_cost.offset, time_samples, predictor)
                     for o in state.dynamic_objects]
    WerlingVisualizer.plot_obstacles(p1, plottable_obs)
    WerlingVisualizer.plot_obstacles(p2, plottable_obs)
    WerlingVisualizer.plot_route(p1, debug.reference_route)
    WerlingVisualizer.plot_route(p2, debug.reference_route)

    WerlingVisualizer.plot_best(p2, debug.trajectories[0])
    WerlingVisualizer.plot_alternatives(p1, debug.trajectories)

    print(debug.costs)

    WerlingVisualizer.plot_route(p1, route_points)

    #fig.show()
    #fig.clear()


