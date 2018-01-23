import time
from unittest.mock import patch

import numpy as np

from decision_making.src.global_constants import EGO_HEIGHT, EGO_WIDTH, EGO_LENGTH, DEFAULT_ACCELERATION, \
    DEFAULT_CURVATURE, TD_STEPS
from decision_making.src.messages.trajectory_parameters import TrajectoryCostParams, SigmoidFunctionParams
from decision_making.src.planning.trajectory.optimal_control.werling_planner import WerlingPlanner
from decision_making.src.planning.types import CURVE_X, CURVE_Y, CURVE_YAW
from decision_making.src.prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.state.state import State, ObjectSize, EgoState, DynamicObject
from decision_making.test.constants import MAP_SERVICE_ABSOLUTE_PATH
from decision_making.test.planning.trajectory.utils import RouteFixture, PlottableSigmoidDynamicBoxObstacle, \
    WerlingVisualizer
from mapping.src.transformations.geometry_utils import CartesianFrame
from mapping.test.model.testable_map_fixtures import map_api_mock
from rte.python.logger.AV_logger import AV_Logger


mock_td_steps = 5


@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
@patch('decision_making.test.planning.trajectory.test_werlingPlanner.TD_STEPS', mock_td_steps)
@patch('decision_making.src.planning.trajectory.optimal_control.werling_planner.TD_STEPS', mock_td_steps)
@patch('decision_making.src.planning.trajectory.optimal_control.werling_planner.SX_STEPS', 2)
@patch('decision_making.src.planning.trajectory.optimal_control.werling_planner.DX_STEPS', 3)
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
    Ts = 1.5

    predictor = RoadFollowingPredictor(logger)

    goal = np.concatenate((route_points[len(route_points) // 2, [CURVE_X, CURVE_Y, CURVE_YAW]], [vT, DEFAULT_ACCELERATION, DEFAULT_CURVATURE]))
    pos1 = np.array([7, -.5])
    yaw1 = 0
    pos2 = np.array([11, 1.5])
    yaw2 = np.pi / 4

    obs = list([
        DynamicObject(obj_id=0, timestamp=950*10e6, x=pos1[0], y=pos1[1], z=0, yaw=yaw1, size=ObjectSize(1.5, 0.5, 0),
                      confidence=1.0, v_x=2.2, v_y=0, acceleration_lon=0.0, omega_yaw=0.0),
        DynamicObject(obj_id=0, timestamp=950*10e6, x=pos2[0], y=pos2[1], z=0, yaw=yaw2, size=ObjectSize(1.5, 0.5, 0),
                      confidence=1.0, v_x=1.1, v_y=0, acceleration_lon=0.0, omega_yaw=0.0)
    ])

    ego = EgoState(obj_id=-1, timestamp=1000*10e6, x=0, y=0, z=0, yaw=0, size=ObjectSize(EGO_LENGTH, EGO_WIDTH, EGO_HEIGHT),
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

    samplable, ctrajectories, costs = planner.plan(state=state, reference_route=route_points[:, :2], goal=goal,
                                                   lon_plan_horizon=Ts, cost_params=cost_params)

    samplable.sample(np.arange(0, 1, 0.01) + ego.timestamp_in_sec)

    assert True

    import matplotlib.pyplot as plt
    plt.switch_backend('QT5Agg')

    fig = plt.figure()
    p1 = fig.add_subplot(211)
    plt.title('A sample from possible trajectories, Ts=%s, TD_STEPS=%s' % (Ts, TD_STEPS))
    p2 = fig.add_subplot(212)
    plt.title('Chosen trajectory')
    time_samples = np.arange(0.0, Ts, 0.1) + ego.timestamp_in_sec
    plottable_obs = [PlottableSigmoidDynamicBoxObstacle(o, cost_params.obstacle_cost.k,
                                                        cost_params.obstacle_cost.offset, time_samples, predictor)
                     for o in state.dynamic_objects]
    WerlingVisualizer.plot_obstacles(p1, plottable_obs)
    WerlingVisualizer.plot_obstacles(p2, plottable_obs)
    WerlingVisualizer.plot_route(p1, route_points[:, :2])
    WerlingVisualizer.plot_route(p2, route_points[:, :2])

    WerlingVisualizer.plot_best(p2, ctrajectories[0])
    WerlingVisualizer.plot_alternatives(p1, ctrajectories)

    print(costs)
    print('\n minimal is: ', np.min(costs))

    WerlingVisualizer.plot_route(p1, route_points)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
    fig.clear()
