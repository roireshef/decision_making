import numpy as np
import time

from decision_making.src.global_constants import OBJECTS_SIGMOID_K_PARAM, LATERAL_SAFETY_MARGIN_FROM_OBJECT, \
    INFINITE_SIGMOID_COST, DEVIATION_FROM_ROAD_COST, DEVIATION_TO_SHOULDER_COST, OUT_OF_LANE_COST, ROAD_SIGMOID_K_PARAM, \
    DEVIATION_FROM_GOAL_LON_COST, DEVIATION_FROM_GOAL_LAT_COST, DEVIATION_FROM_GOAL_MAX_COST, EGO_LENGTH, EGO_WIDTH, \
    LATERAL_SAFETY_MARGIN_FROM_SHOULDER, SHOULDER_SIGMOID_K_PARAM
from decision_making.src.messages.trajectory_parameters import TrajectoryCostParams, SigmoidFunctionParams
from decision_making.src.planning.trajectory.optimal_control.frenet_constraints import FrenetConstraints
from decision_making.src.planning.types import CURVE_X, CURVE_Y, CURVE_YAW, CartesianPoint2D
from decision_making.src.planning.trajectory.optimal_control.werling_planner import WerlingPlanner
from decision_making.src.prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.state.state import State, ObjectSize, EgoState, DynamicObject
from decision_making.test.constants import MAP_SERVICE_ABSOLUTE_PATH
from decision_making.test.planning.trajectory.utils import RouteFixture, PlottableSigmoidDynamicBoxObstacle, \
    WerlingVisualizer, PlottableSigmoidStaticBoxObstacle
from mapping.src.model.constants import ROAD_SHOULDERS_WIDTH
from decision_making.src.planning.utils.math import Math
from mapping.test.model.testable_map_fixtures import testable_map_api
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

    goal = np.concatenate((route_points[len(route_points) // 2, [CURVE_X, CURVE_Y, CURVE_YAW]], [vT]))

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

    fig.show()
    fig.clear()


@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
def test_werlingPlanner_costsShaping():
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

    goal = np.concatenate((route_points[len(route_points) // 2, [CURVE_X, CURVE_Y, CURVE_YAW]], [vT]))

    pos1 = np.array([6, 1.8])
    yaw1 = 0
    pos2 = np.array([20, 5.4])
    yaw2 = np.pi / 16

    obs = list([
        DynamicObject(obj_id=0, timestamp=0, x=pos1[0], y=pos1[1], z=0, yaw=yaw1, size=ObjectSize(4, 1.8, 0),
                      confidence=1.0, v_x=2.2, v_y=0, acceleration_lon=0.0, omega_yaw=0.0),
        DynamicObject(obj_id=0, timestamp=0, x=pos2[0], y=pos2[1], z=0, yaw=yaw2, size=ObjectSize(4, 1.8, 0),
                      confidence=1.0, v_x=1.1, v_y=0, acceleration_lon=0.0, omega_yaw=0.0)
    ])

    ego = EgoState(obj_id=-1, timestamp=0, x=0, y=0, z=0, yaw=0, size=ObjectSize(EGO_LENGTH, EGO_WIDTH, 0),
                   confidence=1.0, v_x=v0, v_y=0, steering_angle=0.0, acceleration_lon=0.0, omega_yaw=0.0)

    state = State(occupancy_state=None, dynamic_objects=obs, ego_state=ego)

    objects_dilation_length = ego.size.length / 2 + LATERAL_SAFETY_MARGIN_FROM_OBJECT
    objects_dilation_width = ego.size.width / 2 + LATERAL_SAFETY_MARGIN_FROM_OBJECT

    reference_route_latitude = 1.8
    road_width = 7.2
    right_shoulder_offset = reference_route_latitude - ego.size.width / 2 + LATERAL_SAFETY_MARGIN_FROM_SHOULDER
    left_shoulder_offset = (road_width - reference_route_latitude) - ego.size.width / 2 + LATERAL_SAFETY_MARGIN_FROM_SHOULDER
    right_road_offset = reference_route_latitude - ego.size.width / 2 + ROAD_SHOULDERS_WIDTH
    left_road_offset = (road_width - reference_route_latitude) - ego.size.width / 2 + ROAD_SHOULDERS_WIDTH

    cost_params = TrajectoryCostParams(
        left_lane_cost=SigmoidFunctionParams(OUT_OF_LANE_COST, ROAD_SIGMOID_K_PARAM, 1.0),
        right_lane_cost=SigmoidFunctionParams(OUT_OF_LANE_COST, ROAD_SIGMOID_K_PARAM, 1.0),
        left_road_cost=SigmoidFunctionParams(DEVIATION_FROM_ROAD_COST, ROAD_SIGMOID_K_PARAM, left_road_offset),
        right_road_cost=SigmoidFunctionParams(DEVIATION_FROM_ROAD_COST, ROAD_SIGMOID_K_PARAM, right_road_offset),
        left_shoulder_cost=SigmoidFunctionParams(DEVIATION_TO_SHOULDER_COST, ROAD_SIGMOID_K_PARAM, left_shoulder_offset),
        right_shoulder_cost=SigmoidFunctionParams(DEVIATION_TO_SHOULDER_COST, ROAD_SIGMOID_K_PARAM, right_shoulder_offset),
        obstacle_cost_x=SigmoidFunctionParams(INFINITE_SIGMOID_COST, OBJECTS_SIGMOID_K_PARAM, objects_dilation_length),
        obstacle_cost_y=SigmoidFunctionParams(INFINITE_SIGMOID_COST, OBJECTS_SIGMOID_K_PARAM, objects_dilation_width),
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

    xrange = (0, 30)
    yrange = (-2, 10)
    x = np.arange(xrange[0], xrange[1], 0.1)
    y = np.arange(yrange[0], yrange[1], 0.1)
    width = x.shape[0]
    height = y.shape[0]
    points = np.array([np.transpose([np.tile(x, y.shape[0]), np.repeat(y, x.shape[0])])])
    goal = np.array([0.9*xrange[1], reference_route_latitude])

    obs_costs = np.zeros(width*height)
    for obj in obs:
        sobj = PlottableSigmoidStaticBoxObstacle(obj, k=OBJECTS_SIGMOID_K_PARAM,
                                                margin=np.array([objects_dilation_length, objects_dilation_width]))
        obs_costs += INFINITE_SIGMOID_COST * sobj.compute_cost_per_point(points)[0]

    latitudes = points[0][:, 1]
    left_shoulder_offsets = (latitudes-reference_route_latitude) - left_shoulder_offset
    right_shoulder_offsets = -(latitudes-reference_route_latitude) - right_shoulder_offset
    left_road_offsets = (latitudes-reference_route_latitude) - left_road_offset
    right_road_offsets = -(latitudes-reference_route_latitude) - right_road_offset
    road_deviations_costs = \
        Math.clipped_sigmoid(left_shoulder_offsets, DEVIATION_TO_SHOULDER_COST, SHOULDER_SIGMOID_K_PARAM) + \
        Math.clipped_sigmoid(right_shoulder_offsets, DEVIATION_TO_SHOULDER_COST, SHOULDER_SIGMOID_K_PARAM) + \
        Math.clipped_sigmoid(left_road_offsets, DEVIATION_FROM_ROAD_COST, ROAD_SIGMOID_K_PARAM) + \
        Math.clipped_sigmoid(right_road_offsets, DEVIATION_FROM_ROAD_COST, ROAD_SIGMOID_K_PARAM)
    goal_deviation_costs = DEVIATION_FROM_GOAL_LON_COST * (points[0][:, 0] - goal[0])**2 + \
                           DEVIATION_FROM_GOAL_LAT_COST * (points[0][:, 1] - goal[1])**2
    goal_deviation_costs = np.clip(goal_deviation_costs, 0, DEVIATION_FROM_GOAL_MAX_COST)

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 14))
    p1 = fig.add_subplot(211)
    p2 = fig.add_subplot(212)
    time_samples = np.arange(0.0, T, 0.1)
    offsets = np.array([cost_params.obstacle_cost_x.offset, cost_params.obstacle_cost_y.offset])
    plottable_obs = [PlottableSigmoidDynamicBoxObstacle(o, cost_params.obstacle_cost_x.k, offsets,
                                                        time_samples, predictor)
                     for o in state.dynamic_objects]
    WerlingVisualizer.plot_obstacles(p1, plottable_obs)
    WerlingVisualizer.plot_obstacles(p2, plottable_obs)
    # WerlingVisualizer.plot_route(p2, debug.reference_route)
    # WerlingVisualizer.plot_best(p2, debug.trajectories[0])

    print(debug.costs)

    p1.plot(np.arange(0, xrange[1]), np.repeat(np.array([0]), xrange[1]), '-k')
    p1.plot(np.arange(0, xrange[1]), np.repeat(np.array([road_width]), xrange[1]), '-k')
    p1.plot(np.arange(0, xrange[1]), np.repeat(np.array([-1.5]), xrange[1]), '-r')
    p1.plot(np.arange(0, xrange[1]), np.repeat(np.array([road_width+1.5]), xrange[1]), '-r')
    p1.plot(np.arange(0, xrange[1]), np.repeat(np.array([road_width/2]), xrange[1]), '--w')

    x = points[0, :, 0].reshape(height, width)
    y = points[0, :, 1].reshape(height, width)
    z = obs_costs.reshape(height, width) + road_deviations_costs.reshape(height, width) \
        #+ goal_deviation_costs.reshape(height, width)
    #z = np.log(z)
    z = np.clip(z, 0, 5000)
    p1.contourf(x, y, z, 100)


    p2.plot(np.arange(0, xrange[1]), np.repeat(np.array([0]), xrange[1]), '-k')
    p2.plot(np.arange(0, xrange[1]), np.repeat(np.array([road_width]), xrange[1]), '-k')
    p2.plot(np.arange(0, xrange[1]), np.repeat(np.array([-1.5]), xrange[1]), '-r')
    p2.plot(np.arange(0, xrange[1]), np.repeat(np.array([road_width+1.5]), xrange[1]), '-r')
    p2.plot(np.arange(0, xrange[1]), np.repeat(np.array([road_width/2]), xrange[1]), '--w')

    z = obs_costs.reshape(height, width) + road_deviations_costs.reshape(height, width) \
        #+ goal_deviation_costs.reshape(height, width)
    #z = np.log(z)
    z = np.clip(z, 0., 200.)
    p2.contourf(x, y, z, 100)

    #for i, p in enumerate(points[0]):
    #    p1.plot(p[0], p[1], costs[0][i])

    fig.show()
    fig.clear()
