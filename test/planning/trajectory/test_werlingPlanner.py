import numpy as np
import time

from decision_making.src.global_constants import OBJECTS_SIGMOID_K_PARAM, LATERAL_SAFETY_MARGIN_FROM_OBJECT, \
    INFINITE_SIGMOID_COST, DEVIATION_FROM_ROAD_COST, DEVIATION_TO_SHOULDER_COST, OUT_OF_LANE_COST, ROAD_SIGMOID_K_PARAM, \
    DEVIATION_FROM_GOAL_LON_COST, DEVIATION_FROM_GOAL_LAT_COST, EGO_LENGTH, EGO_WIDTH, \
    SHOULDER_SIGMOID_OFFSET, SHOULDER_SIGMOID_K_PARAM, VELOCITY_LIMITS, \
    LON_ACCELERATION_LIMITS, DEFAULT_ACCELERATION, DEFAULT_CURVATURE, EGO_HEIGHT, LON_JERK_COST, LAT_JERK_COST
from decision_making.src.messages.trajectory_parameters import TrajectoryCostParams, SigmoidFunctionParams
from decision_making.src.planning.trajectory.cost_function import Jerk
from decision_making.src.planning.trajectory.optimal_control.frenet_constraints import FrenetConstraints
from decision_making.src.planning.types import CURVE_X, CURVE_Y, CURVE_YAW, CartesianPoint2D, C_Y, \
    CartesianExtendedTrajectory, C_X, C_YAW, C_V
from decision_making.src.planning.trajectory.optimal_control.werling_planner import WerlingPlanner
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
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
from mapping.src.model.constants import ROAD_SHOULDERS_WIDTH

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
                                       obstacle_cost_x=SigmoidFunctionParams(100, 10.0, 0.3),
                                       obstacle_cost_y=SigmoidFunctionParams(100, 10.0, 0.3),
                                       lon_jerk_cost=LON_JERK_COST,
                                       lat_jerk_cost=LAT_JERK_COST,
                                       dist_from_goal_lat_sq_cost=1.0,
                                       dist_from_goal_lon_sq_cost=1.0,
                                       velocity_limits=np.array([v_min, v_max]),
                                       acceleration_limits=np.array([a_min, a_max]))

    planner = WerlingPlanner(logger, predictor)

    start_time = time.time()

    samplable, ctrajectories, costs, _ = planner.plan(state=state, reference_route=route_points[:, :2], goal=goal,
                                       goal_time=ego.timestamp_in_sec + T, cost_params=cost_params)

    samplable.sample(np.arange(0, 1, 0.1) + ego.timestamp_in_sec)

    end_time = time.time() - start_time

    assert True

    import matplotlib.pyplot as plt

    fig = plt.figure()
    p1 = fig.add_subplot(211)
    p2 = fig.add_subplot(212)
    time_samples = np.arange(0.0, T, 0.1) + ego.timestamp_in_sec
    plottable_obs = [PlottableSigmoidDynamicBoxObstacle(o, cost_params.obstacle_cost_x.k,
                                                        np.array([cost_params.obstacle_cost_x.offset,
                                                                  cost_params.obstacle_cost_y.offset]),
                                                        time_samples, predictor)
                     for o in state.dynamic_objects]
    WerlingVisualizer.plot_obstacles(p1, plottable_obs)
    WerlingVisualizer.plot_obstacles(p2, plottable_obs)
    WerlingVisualizer.plot_route(p1, route_points[:, :2])
    WerlingVisualizer.plot_route(p2, route_points[:, :2])

    WerlingVisualizer.plot_best(p2, ctrajectories[0])
    WerlingVisualizer.plot_alternatives(p1, ctrajectories, costs)

    print(costs)

    WerlingVisualizer.plot_route(p1, route_points)

    fig.show()
    fig.clear()


@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
def test_werlingPlanner_twoStaticObjScenario_withCostViz():
    logger = AV_Logger.get_logger('test_werlingPlanner_twoStaticObjScenario_withCostViz')

    lane_width = 3.6
    num_lanes = 2
    road_width = num_lanes*lane_width
    reference_route_latitude = 3*lane_width/2

    lng = 40
    step = 0.2
    curvature = 0.2

    route_xy = RouteFixture.get_cubic_route(lng=lng, lat=reference_route_latitude, ext=0, step=step, curvature=curvature)
    ext = 4
    ext_route_xy = RouteFixture.get_cubic_route(lng=lng, lat=reference_route_latitude, ext=ext, step=step, curvature=curvature)
    route_points = CartesianFrame.add_yaw_and_derivatives(route_xy)
    ext_route_points = CartesianFrame.add_yaw_and_derivatives(ext_route_xy)

    start_latitude = lane_width/2
    goal_latitude = reference_route_latitude
    target_lane = int(goal_latitude/lane_width)

    xrange = (route_points[0, C_X], route_points[-1, C_X])
    yrange = (np.min(route_points[:, C_Y]) - reference_route_latitude - ROAD_SHOULDERS_WIDTH,
              np.max(route_points[:, C_Y]) - reference_route_latitude + road_width + ROAD_SHOULDERS_WIDTH)
    x = np.arange(xrange[0], xrange[1], 0.1)
    y = np.arange(yrange[0], yrange[1], 0.1)
    width = x.shape[0]
    height = y.shape[0]
    points = np.array([np.transpose([np.tile(x, y.shape[0]), np.repeat(y, x.shape[0])])])

    frenet = FrenetSerret2DFrame(ext_route_points[:, :2])
    pos_x = points[0].reshape(height, width, 2)
    s_x, a_r, _, N_r, _, _ = frenet._project_cartesian_points(pos_x)
    d_x = np.einsum('tpi,tpi->tp', pos_x - a_r, N_r)
    fpoints = np.c_[s_x.flatten()-s_x.flatten()[0], d_x.flatten()]

    v0 = 6
    vT = 10
    T = 4.6

    ftraj_start_goal = np.array([np.array([ext, v0, 0, start_latitude-reference_route_latitude, 0, 0]),
                                 np.array([lng+ext, vT, 0, goal_latitude-reference_route_latitude, 0, 0])])
    ctraj_start_goal = frenet.ftrajectory_to_ctrajectory(ftraj_start_goal)

    predictor = RoadFollowingPredictor(logger)

    goal = ctraj_start_goal[1]
        #np.concatenate((route_points[-1, [CURVE_X, CURVE_Y, CURVE_YAW]], [vT, DEFAULT_ACCELERATION, DEFAULT_CURVATURE]))
    goal[C_X] -= 0.001
    goal[C_Y] += goal_latitude - reference_route_latitude

    pos1 = np.array([6, 5.4])
    yaw1 = 0
    pos2 = np.array([22, 3.0])
    yaw2 = np.pi / 32

    obs = list([
        DynamicObject(obj_id=0, timestamp=0, x=pos1[0], y=pos1[1], z=0, yaw=yaw1, size=ObjectSize(4, 1.8, 0),
                      confidence=1.0, v_x=0, v_y=0, acceleration_lon=0.0, omega_yaw=0.0),
        DynamicObject(obj_id=0, timestamp=0, x=pos2[0], y=pos2[1], z=0, yaw=yaw2, size=ObjectSize(4, 1.8, 0),
                      confidence=1.0, v_x=0, v_y=0, acceleration_lon=0.0, omega_yaw=0.0)
    ])
    obs = list([])

    ego = EgoState(obj_id=-1, timestamp=0, x=ctraj_start_goal[0][C_X], y=ctraj_start_goal[0][C_Y], z=0,
                   yaw=ctraj_start_goal[0][C_YAW],  # route_points[0, CURVE_YAW],
                   size=ObjectSize(EGO_LENGTH, EGO_WIDTH, 0),
                   confidence=1.0, v_x=ctraj_start_goal[0][C_V], v_y=0, steering_angle=0.0, acceleration_lon=0.0, omega_yaw=0.0)

    state = State(occupancy_state=None, dynamic_objects=obs, ego_state=ego)

    objects_dilation_length = ego.size.length / 2 + LATERAL_SAFETY_MARGIN_FROM_OBJECT
    objects_dilation_width = ego.size.width / 2 + LATERAL_SAFETY_MARGIN_FROM_OBJECT
    right_lane_offset = max(0.0, reference_route_latitude - ego.size.width / 2 - target_lane * lane_width)
    left_lane_offset = (road_width - reference_route_latitude) - ego.size.width / 2 - (num_lanes - target_lane - 1) * lane_width
    right_shoulder_offset = reference_route_latitude - ego.size.width / 2 + SHOULDER_SIGMOID_OFFSET
    left_shoulder_offset = (road_width - reference_route_latitude) - ego.size.width / 2 + SHOULDER_SIGMOID_OFFSET
    right_road_offset = reference_route_latitude - ego.size.width / 2 + ROAD_SHOULDERS_WIDTH
    left_road_offset = (road_width - reference_route_latitude) - ego.size.width / 2 + ROAD_SHOULDERS_WIDTH

    cost_params = TrajectoryCostParams(
        left_lane_cost=SigmoidFunctionParams(OUT_OF_LANE_COST, ROAD_SIGMOID_K_PARAM, left_lane_offset),
        right_lane_cost=SigmoidFunctionParams(OUT_OF_LANE_COST, ROAD_SIGMOID_K_PARAM, right_lane_offset),
        left_road_cost=SigmoidFunctionParams(DEVIATION_FROM_ROAD_COST, ROAD_SIGMOID_K_PARAM, left_road_offset),
        right_road_cost=SigmoidFunctionParams(DEVIATION_FROM_ROAD_COST, ROAD_SIGMOID_K_PARAM, right_road_offset),
        left_shoulder_cost=SigmoidFunctionParams(DEVIATION_TO_SHOULDER_COST, ROAD_SIGMOID_K_PARAM, left_shoulder_offset),
        right_shoulder_cost=SigmoidFunctionParams(DEVIATION_TO_SHOULDER_COST, ROAD_SIGMOID_K_PARAM, right_shoulder_offset),
        obstacle_cost_x=SigmoidFunctionParams(INFINITE_SIGMOID_COST, OBJECTS_SIGMOID_K_PARAM, objects_dilation_length),
        obstacle_cost_y=SigmoidFunctionParams(INFINITE_SIGMOID_COST, OBJECTS_SIGMOID_K_PARAM, objects_dilation_width),
        lon_jerk_cost=LON_JERK_COST,
        lat_jerk_cost=LAT_JERK_COST,
        dist_from_goal_lat_sq_cost=DEVIATION_FROM_GOAL_LAT_COST,
        dist_from_goal_lon_sq_cost=DEVIATION_FROM_GOAL_LON_COST,
        velocity_limits=VELOCITY_LIMITS,
        acceleration_limits=LON_ACCELERATION_LIMITS)

    planner = WerlingPlanner(logger, predictor)

    start_time = time.time()

    samplable, ctrajectories, costs, partial_costs = planner.plan(state=state, reference_route=ext_route_points[:, :2],
                                                                  goal=goal, goal_time=T, cost_params=cost_params)

    end_time = time.time() - start_time

    obs_costs = np.zeros(width * height)
    for obj in obs:
        sobj = PlottableSigmoidStaticBoxObstacle(obj, k=OBJECTS_SIGMOID_K_PARAM,
                                                 margin=np.array([objects_dilation_length, objects_dilation_width]))
        obs_costs += INFINITE_SIGMOID_COST * sobj.compute_cost_per_point(points)[0]

    latitudes = fpoints[:, 1]
    left_lane_offsets = latitudes - left_lane_offset
    right_lane_offsets = -latitudes - right_lane_offset
    left_shoulder_offsets = latitudes - left_shoulder_offset
    right_shoulder_offsets = -latitudes - right_shoulder_offset
    left_road_offsets = latitudes - left_road_offset
    right_road_offsets = -latitudes - right_road_offset

    road_deviations_costs = \
        Math.clipped_sigmoid(left_lane_offsets, OUT_OF_LANE_COST, SHOULDER_SIGMOID_K_PARAM) + \
        Math.clipped_sigmoid(right_lane_offsets, OUT_OF_LANE_COST, SHOULDER_SIGMOID_K_PARAM) + \
        Math.clipped_sigmoid(left_shoulder_offsets, DEVIATION_TO_SHOULDER_COST, SHOULDER_SIGMOID_K_PARAM) + \
        Math.clipped_sigmoid(right_shoulder_offsets, DEVIATION_TO_SHOULDER_COST, SHOULDER_SIGMOID_K_PARAM) + \
        Math.clipped_sigmoid(left_road_offsets, DEVIATION_FROM_ROAD_COST, ROAD_SIGMOID_K_PARAM) + \
        Math.clipped_sigmoid(right_road_offsets, DEVIATION_FROM_ROAD_COST, ROAD_SIGMOID_K_PARAM)
    goal_deviation_costs = DEVIATION_FROM_GOAL_LON_COST * (points[0][:, 0] - goal[0])**2 + \
                           DEVIATION_FROM_GOAL_LAT_COST * (points[0][:, 1] - goal[1])**2

    import matplotlib.pyplot as plt

    fig = plt.figure()
    p1 = fig.add_subplot(211)
    p2 = fig.add_subplot(212)
    time_samples = np.arange(0.0, T, 0.1)
    offsets = np.array([cost_params.obstacle_cost_x.offset, cost_params.obstacle_cost_y.offset])
    plottable_obs = [PlottableSigmoidDynamicBoxObstacle(o, cost_params.obstacle_cost_x.k, offsets, time_samples,
                                                        predictor)
                     for o in state.dynamic_objects]

    x = points[0, :, 0].reshape(height, width)
    y = points[0, :, 1].reshape(height, width)
    z = obs_costs.reshape(height, width) + road_deviations_costs.reshape(height, width) \
        #+ goal_deviation_costs.reshape(height, width)

    diff = np.diff(route_points[:, :2], axis=0)
    angles = np.arctan2(diff[:, 1], diff[:, 0])
    angles = np.concatenate((angles, np.array([angles[-1]])))

    for p in list([p1, p2]):
        WerlingVisualizer.plot_obstacles(p, plottable_obs)
        WerlingVisualizer.plot_obstacles(p, plottable_obs)
        WerlingVisualizer.plot_route(p, route_points[:, :2])
        d = reference_route_latitude
        WerlingVisualizer.plot_route(p, np.c_[route_points[:, 0] + d*np.sin(angles), route_points[:, 1] - d*np.cos(angles)], '-k')
        d = road_width - reference_route_latitude
        WerlingVisualizer.plot_route(p, np.c_[route_points[:, 0] - d*np.sin(angles), route_points[:, 1] + d*np.cos(angles)], '-k')
        d = road_width/2 - reference_route_latitude
        WerlingVisualizer.plot_route(p, np.c_[route_points[:, 0] - d*np.sin(angles), route_points[:, 1] + d*np.cos(angles)], '--w')

        # plot ego's best position for both obstacles
        for obs in state.dynamic_objects:
            min_cost_y = np.argmin(z[:, int((obs.x - xrange[0])/0.1)]) * 0.1 + yrange[0]
            p.plot(np.arange(- ego.size.length / 2, ego.size.length / 2 + 0.01) + obs.x,
                   np.repeat(np.array([min_cost_y+ego.size.width/2]), np.ceil(ego.size.length)+1), '*w')
            p.plot(np.arange(- ego.size.length / 2, ego.size.length / 2 + 0.01) + obs.x,
                   np.repeat(np.array([min_cost_y - ego.size.width / 2]), np.ceil(ego.size.length)+1), '*w')
            p.plot(np.repeat(np.array([-ego.size.length / 2 + obs.x]), np.ceil(ego.size.width)+1),
                   np.arange(min_cost_y-ego.size.width/2, min_cost_y+ego.size.width/2 + 0.01), '*w')
            p.plot(np.repeat(np.array([ego.size.length / 2 + obs.x]), np.ceil(ego.size.width)+1),
                   np.arange(min_cost_y-ego.size.width/2, min_cost_y+ego.size.width/2 + 0.01), '*w')

    d = reference_route_latitude + ROAD_SHOULDERS_WIDTH
    WerlingVisualizer.plot_route(p2, np.c_[route_points[:, 0] + d*np.sin(angles), route_points[:, 1] - d*np.cos(angles)], '-r')
    d = road_width - reference_route_latitude + ROAD_SHOULDERS_WIDTH
    WerlingVisualizer.plot_route(p2, np.c_[route_points[:, 0] - d*np.sin(angles), route_points[:, 1] + d*np.cos(angles)], '-r')

    z = np.log(1 + z)
    p2.contourf(x, y, z, 100)

    WerlingVisualizer.plot_best(p2, ctrajectories[0])
    WerlingVisualizer.plot_alternatives(p1, ctrajectories, costs)

    WerlingVisualizer.plot_goal(p2, goal)

    print(partial_costs)

    fig.show()
    fig.clear()


# @patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
# def test_werlingPlanner_costsShaping():
#     logger = AV_Logger.get_logger('test_werlingPlanner_toyScenario_noException')
#     route_points = CartesianFrame.add_yaw_and_derivatives(
#         RouteFixture.get_route(lng=10, k=1, step=1, lat=3, offset=-.5))
#
#     v0 = 6
#     vT = 8
#     T = 1.5
#
#     predictor = RoadFollowingPredictor(logger)
#
#     goal = np.concatenate((route_points[len(route_points) // 2, [CURVE_X, CURVE_Y, CURVE_YAW]], [vT, DEFAULT_ACCELERATION, DEFAULT_CURVATURE]))
#
#     pos1 = np.array([6, 1.8])
#     yaw1 = 0
#     pos2 = np.array([20, 2.7])
#     yaw2 = 0
#     pos3 = np.array([35, 3.5])
#     yaw3 = 0  # np.pi/16
#
#     obs = list([
#         DynamicObject(obj_id=0, timestamp=0, x=pos1[0], y=pos1[1], z=0, yaw=yaw1, size=ObjectSize(4, 1.8, 0),
#                       confidence=1.0, v_x=2.2, v_y=0, acceleration_lon=0.0, omega_yaw=0.0),
#         DynamicObject(obj_id=0, timestamp=0, x=pos2[0], y=pos2[1], z=0, yaw=yaw2, size=ObjectSize(4, 1.8, 0),
#                       confidence=1.0, v_x=1.1, v_y=0, acceleration_lon=0.0, omega_yaw=0.0),
#         DynamicObject(obj_id=0, timestamp=0, x=pos3[0], y=pos3[1], z=0, yaw=yaw3, size=ObjectSize(4, 1.8, 0),
#                       confidence=1.0, v_x=1.1, v_y=0, acceleration_lon=0.0, omega_yaw=0.0)
#     ])
#
#     ego = EgoState(obj_id=-1, timestamp=0, x=0, y=0, z=0, yaw=0, size=ObjectSize(EGO_LENGTH, EGO_WIDTH, 0),
#                    confidence=1.0, v_x=v0, v_y=0, steering_angle=0.0, acceleration_lon=0.0, omega_yaw=0.0)
#
#     state = State(occupancy_state=None, dynamic_objects=obs, ego_state=ego)
#
#     objects_dilation_length = ego.size.length / 2 + LATERAL_SAFETY_MARGIN_FROM_OBJECT
#     objects_dilation_width = ego.size.width / 2 + LATERAL_SAFETY_MARGIN_FROM_OBJECT
#
#     lane_width = 3.6
#     num_lanes = 2
#     road_width = num_lanes*lane_width
#     reference_route_latitude = 1.8
#     target_lane = int(reference_route_latitude/lane_width)
#
#     right_lane_offset = target_lane*lane_width - ego.size.width / 2
#     left_lane_offset = (target_lane+1)*lane_width - ego.size.width / 2
#     right_shoulder_offset = -ego.size.width / 2 + SHOULDER_SIGMOID_OFFSET
#     left_shoulder_offset = road_width - ego.size.width / 2 + SHOULDER_SIGMOID_OFFSET
#     right_road_offset = -ego.size.width / 2 + ROAD_SHOULDERS_WIDTH
#     left_road_offset = road_width - ego.size.width / 2 + ROAD_SHOULDERS_WIDTH
#
#     cost_params = TrajectoryCostParams(
#         left_lane_cost=SigmoidFunctionParams(OUT_OF_LANE_COST, ROAD_SIGMOID_K_PARAM, left_lane_offset),
#         right_lane_cost=SigmoidFunctionParams(OUT_OF_LANE_COST, ROAD_SIGMOID_K_PARAM, right_lane_offset),
#         left_road_cost=SigmoidFunctionParams(DEVIATION_FROM_ROAD_COST, ROAD_SIGMOID_K_PARAM, left_road_offset),
#         right_road_cost=SigmoidFunctionParams(DEVIATION_FROM_ROAD_COST, ROAD_SIGMOID_K_PARAM, right_road_offset),
#         left_shoulder_cost=SigmoidFunctionParams(DEVIATION_TO_SHOULDER_COST, ROAD_SIGMOID_K_PARAM, left_shoulder_offset),
#         right_shoulder_cost=SigmoidFunctionParams(DEVIATION_TO_SHOULDER_COST, ROAD_SIGMOID_K_PARAM, right_shoulder_offset),
#         obstacle_cost_x=SigmoidFunctionParams(INFINITE_SIGMOID_COST, OBJECTS_SIGMOID_K_PARAM, objects_dilation_length),
#         obstacle_cost_y=SigmoidFunctionParams(INFINITE_SIGMOID_COST, OBJECTS_SIGMOID_K_PARAM, objects_dilation_width),
#         lon_jerk_cost = LON_JERK_COST,
#         lat_jerk_cost = LAT_JERK_COST,
#         dist_from_goal_lat_sq_cost=DEVIATION_FROM_GOAL_LAT_COST,
#         dist_from_goal_lon_sq_cost=DEVIATION_FROM_GOAL_LON_COST,
#         velocity_limits=VELOCITY_LIMITS,
#         acceleration_limits=LON_ACCELERATION_LIMITS)
#
#     planner = WerlingPlanner(logger, predictor)
#
#     start_time = time.time()
#
#     _, _, debug = planner.plan(state=state, reference_route=route_points[:, :2], goal=goal,
#                                goal_time=T, cost_params=cost_params)
#
#     end_time = time.time() - start_time
#
#     xrange = (0, 60)
#     yrange = (-2, 10)
#     x = np.arange(xrange[0], xrange[1], 0.1)
#     y = np.arange(yrange[0], yrange[1], 0.1)
#     width = x.shape[0]
#     height = y.shape[0]
#     points = np.array([np.transpose([np.tile(x, y.shape[0]), np.repeat(y, x.shape[0])])])
#
#     goal = np.array([0.9*xrange[1], reference_route_latitude])
#
#     obs_costs = np.zeros(width*height)
#     for obj in obs:
#         sobj = PlottableSigmoidStaticBoxObstacle(obj, k=OBJECTS_SIGMOID_K_PARAM,
#                                                 margin=np.array([objects_dilation_length, objects_dilation_width]))
#         obs_costs += INFINITE_SIGMOID_COST * sobj.compute_cost_per_point(points)[0]
#
#     latitudes = points[0][:, 1]
#     left_lane_offsets = latitudes - left_lane_offset
#     right_lane_offsets = -latitudes - right_lane_offset
#     left_shoulder_offsets = latitudes - left_shoulder_offset
#     right_shoulder_offsets = -latitudes - right_shoulder_offset
#     left_road_offsets = latitudes - left_road_offset
#     right_road_offsets = -latitudes - right_road_offset
#
#     road_deviations_costs = \
#         Math.clipped_sigmoid(left_lane_offsets, OUT_OF_LANE_COST, SHOULDER_SIGMOID_K_PARAM) + \
#         Math.clipped_sigmoid(right_lane_offsets, OUT_OF_LANE_COST, SHOULDER_SIGMOID_K_PARAM) + \
#         Math.clipped_sigmoid(left_shoulder_offsets, DEVIATION_TO_SHOULDER_COST, SHOULDER_SIGMOID_K_PARAM) + \
#         Math.clipped_sigmoid(right_shoulder_offsets, DEVIATION_TO_SHOULDER_COST, SHOULDER_SIGMOID_K_PARAM) + \
#         Math.clipped_sigmoid(left_road_offsets, DEVIATION_FROM_ROAD_COST, ROAD_SIGMOID_K_PARAM) + \
#         Math.clipped_sigmoid(right_road_offsets, DEVIATION_FROM_ROAD_COST, ROAD_SIGMOID_K_PARAM)
#     goal_deviation_costs = DEVIATION_FROM_GOAL_LON_COST * (points[0][:, 0] - goal[0])**2 + \
#                            DEVIATION_FROM_GOAL_LAT_COST * (points[0][:, 1] - goal[1])**2
#     #goal_deviation_costs = np.clip(goal_deviation_costs, 0, DEVIATION_FROM_GOAL_MAX_COST)
#
#     import matplotlib.pyplot as plt
#
#     fig = plt.figure(figsize=(12, 18))
#     p1 = fig.add_subplot(211)
#     p2 = fig.add_subplot(212)
#     time_samples = np.arange(0.0, T, 0.1)
#     offsets = np.array([cost_params.obstacle_cost_x.offset, cost_params.obstacle_cost_y.offset])
#     plottable_obs = [PlottableSigmoidDynamicBoxObstacle(o, cost_params.obstacle_cost_x.k, offsets,
#                                                         time_samples, predictor)
#                      for o in state.dynamic_objects]
#     # WerlingVisualizer.plot_obstacles(p1, plottable_obs)
#     # WerlingVisualizer.plot_obstacles(p2, plottable_obs)
#     # WerlingVisualizer.plot_route(p2, debug.reference_route)
#     # WerlingVisualizer.plot_best(p2, debug.trajectories[0])
#
#     print(debug.costs)
#
#     x = points[0, :, 0].reshape(height, width)
#     y = points[0, :, 1].reshape(height, width)
#     z = obs_costs.reshape(height, width) + road_deviations_costs.reshape(height, width) \
#         #+ goal_deviation_costs.reshape(height, width)
#
#     for p in list([p1, p2]):
#         WerlingVisualizer.plot_obstacles(p, plottable_obs)
#         p.plot(np.arange(0, xrange[1]), np.repeat(np.array([0]), xrange[1]), '-k')
#         p.plot(np.arange(0, xrange[1]), np.repeat(np.array([road_width]), xrange[1]), '-k')
#         p.plot(np.arange(0, xrange[1]), np.repeat(np.array([-1.5]), xrange[1]), '-r')
#         p.plot(np.arange(0, xrange[1]), np.repeat(np.array([road_width+1.5]), xrange[1]), '-r')
#         p.plot(np.arange(0, xrange[1]), np.repeat(np.array([road_width/2]), xrange[1]), '--w')
#
#         # plot ego's best position for both obstacles
#         for obs in state.dynamic_objects:
#             min_cost_y = np.argmin(z[:, int((obs.x - xrange[0])/0.1)]) * 0.1 + yrange[0]
#             p.plot(np.arange(- ego.size.length / 2, ego.size.length / 2 + 0.01) + obs.x,
#                    np.repeat(np.array([min_cost_y+ego.size.width/2]), np.ceil(ego.size.length)+1), '*w')
#             p.plot(np.arange(- ego.size.length / 2, ego.size.length / 2 + 0.01) + obs.x,
#                    np.repeat(np.array([min_cost_y - ego.size.width / 2]), np.ceil(ego.size.length)+1), '*w')
#             p.plot(np.repeat(np.array([-ego.size.length / 2 + obs.x]), np.ceil(ego.size.width)+1),
#                    np.arange(min_cost_y-ego.size.width/2, min_cost_y+ego.size.width/2 + 0.01), '*w')
#             p.plot(np.repeat(np.array([ego.size.length / 2 + obs.x]), np.ceil(ego.size.width)+1),
#                    np.arange(min_cost_y-ego.size.width/2, min_cost_y+ego.size.width/2 + 0.01), '*w')
#
#     min_cost_y = np.argmin(z[:, int(50 / 0.1)]) * 0.1 + yrange[0]
#     p.plot(np.arange(- ego.size.length / 2, ego.size.length / 2 + 0.01) + 50,
#            np.repeat(np.array([min_cost_y + ego.size.width / 2]), np.ceil(ego.size.length) + 1), '*w')
#     p.plot(np.arange(- ego.size.length / 2, ego.size.length / 2 + 0.01) + 50,
#            np.repeat(np.array([min_cost_y - ego.size.width / 2]), np.ceil(ego.size.length) + 1), '*w')
#     p.plot(np.repeat(np.array([-ego.size.length / 2 + 50]), np.ceil(ego.size.width) + 1),
#            np.arange(min_cost_y - ego.size.width / 2, min_cost_y + ego.size.width / 2 + 0.01), '*w')
#     p.plot(np.repeat(np.array([ego.size.length / 2 + 50]), np.ceil(ego.size.width) + 1),
#            np.arange(min_cost_y - ego.size.width / 2, min_cost_y + ego.size.width / 2 + 0.01), '*w')
#
#     # z = np.clip(z, 0, 5000)
#     p1.contourf(x, y, z, 100)
#
#     z = np.log(1+z)
#     # z = np.clip(z, 0., 1000.)
#     p2.contourf(x, y, z, 100)
#
#     #for i, p in enumerate(points[0]):
#     #    p1.plot(p[0], p[1], costs[0][i])
#
#     fig.show()
#     fig.clear()

@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
def test_calcJerkCosts():
    # [C_X, C_Y, C_YAW, C_V, C_A, C_K]
    ctraj: CartesianExtendedTrajectory = np.array([
        np.array([0, 0, 0,   1, 0, 0]),
        np.array([0, 0, 0, 1.1, 1, 1]),
        np.array([0, 0, 0, 1.3, 3, 2])
    ])
    lon_jerks, lat_jerks = Jerk.compute_jerks(np.array([ctraj]), None, 0.1)
    assert np.isclose(lon_jerks[0], 500) and np.isclose(lat_jerks[0], 617.3)
