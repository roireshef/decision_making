from decision_making.src.scene.scene_static_model import SceneStaticModel
from decision_making.test.messages.scene_static_fixture import scene_static_pg_split
from logging import Logger

import numpy as np

from decision_making.src.global_constants import EPS, OBSTACLE_SIGMOID_COST, OBSTACLE_SIGMOID_K_PARAM, \
    LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT, LATERAL_SAFETY_MARGIN_FROM_OBJECT_FOR_TP_COST, BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED
from decision_making.src.messages.trajectory_parameters import TrajectoryCostParams, SigmoidFunctionParams
from decision_making.src.planning.trajectory.cost_function import TrajectoryPlannerCosts
from decision_making.src.planning.trajectory.samplable_werling_trajectory import SamplableWerlingTrajectory
from decision_making.src.planning.types import FS_DX
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GeneralizedFrenetSerretFrame, \
    FrenetSubSegment
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.state.map_state import MapState
from decision_making.src.state.state import ObjectSize, DynamicObject, State, EgoState
from decision_making.src.utils.map_utils import MapUtils


def test_computeObstacleCosts_threeSRoutesOneObstacle_validScore(scene_static_pg_split):
    """
    Test TP obstacle cost.
    Ego has 3 lane-change trajectories (from right to left) with same init/end constraints but different T_d: [4, 6, 8].
    Dynamic object is static, located on the right lane, such that ego reaches it longitudinally at time t = 4.
    The result:
    The first trajectory is collision-free.
    The second trajectory is too close to the object.
    The third trajectory collides with the object.
    """
    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_split)

    logger = Logger("test_computeCost_threeSRoutesOneObstacle_validScore")
    road_id = 20
    lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_id)[0]
    lane_width = MapUtils.get_lane_width(lane_id, s=0)
    s = 0
    init_d = 0
    end_d = init_d + lane_width
    v = 10
    T = 8
    T_d = np.array([T-4, T-2, T])
    # Ego has 3 lane-change trajectories (from right to left) with same init/end constraints but different
    # T_d = [4, 6, 8]. Ego moves with constant longitudinal velocity 10 m/s during 8 seconds.
    init_fstate = np.array([s, v, 0, init_d, 0, 0])
    init_fstates = np.tile(init_fstate, 3).reshape(3, 6)
    target_fstate = np.array([s + T * v, v, 0, end_d, 0, 0])
    target_fstates = np.tile(target_fstate, 3).reshape(3, 6)

    # Dynamic object is static, located on the right lane, such that ego reaches it longitudinally at time t = 4.
    obj_map_state = MapState(np.array([s + T_d[0] * v, EPS, 0, init_d, 0, 0]), lane_id)
    obj_size = ObjectSize(4, 1.8, 0)
    time_points = np.arange(0, T + EPS, 0.1)

    # create State
    ego_map_state = MapState(init_fstate, lane_id)
    ego = EgoState.create_from_map_state(0, 0, ego_map_state, obj_size, 0, off_map=False)
    obj = DynamicObject.create_from_map_state(1, 0, obj_map_state, obj_size, 0, off_map=False)
    state = State(False, None, [obj], ego)

    # calculate polynomials for s & d
    constraints_s = np.concatenate((init_fstates[:, :FS_DX], target_fstates[:, :FS_DX]), axis=-1)
    constraints_d = np.concatenate((init_fstates[:, FS_DX:], target_fstates[:, FS_DX:]), axis=-1)

    A_s_inv = np.linalg.inv(QuinticPoly1D.time_constraints_matrix(T))
    poly_coefs_s = QuinticPoly1D.solve(A_s_inv, constraints_s)
    A_d_inv = np.linalg.inv(QuinticPoly1D.time_constraints_tensor(T_d))
    poly_coefs_d = QuinticPoly1D.zip_solve(A_d_inv, constraints_d)

    frenet_frame = MapUtils.get_lane_frenet_frame(ego.map_state.lane_id)
    sub_segment = FrenetSubSegment(ego.map_state.lane_id, 0, frenet_frame.s_max)
    reference_route = GeneralizedFrenetSerretFrame.build([frenet_frame], [sub_segment])

    # create 3 ctrajectories
    ctrajectories = []
    for i in range(poly_coefs_s.shape[0]):
        samplable_trajectory = SamplableWerlingTrajectory(timestamp_in_sec=ego.timestamp_in_sec,
                                                          T_s=T, T_d=T_d[i], T_extended=T,
                                                          frenet_frame=frenet_frame,
                                                          poly_s_coefs=poly_coefs_s[i], poly_d_coefs=poly_coefs_d[i])
        ctrajectory = samplable_trajectory.sample(time_points)
        ctrajectories.append(ctrajectory)
    ctrajectories = np.array(ctrajectories)

    # calculate obstacle costs for each trajectory
    predictor = RoadFollowingPredictor(logger)
    objects_cost_x = SigmoidFunctionParams(w=OBSTACLE_SIGMOID_COST, k=OBSTACLE_SIGMOID_K_PARAM,
                                           offset=LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT)  # Very high (inf) cost
    objects_cost_y = SigmoidFunctionParams(w=OBSTACLE_SIGMOID_COST, k=OBSTACLE_SIGMOID_K_PARAM,
                                           offset=LATERAL_SAFETY_MARGIN_FROM_OBJECT_FOR_TP_COST)  # Very high (inf) cost
    cost_params = TrajectoryCostParams(obstacle_cost_x=objects_cost_x, obstacle_cost_y=objects_cost_y,
                                       left_lane_cost=None, right_lane_cost=None, left_shoulder_cost=None,
                                       right_shoulder_cost=None, left_road_cost=None, right_road_cost=None,
                                       dist_from_goal_cost=None, dist_from_goal_lat_factor=None, lon_jerk_cost_weight=None,
                                       lat_jerk_cost_weight=None, velocity_limits=None, lon_acceleration_limits=None,
                                       lat_acceleration_limits=None,
                                       desired_velocity=BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED)
    pointwise_costs = TrajectoryPlannerCosts.compute_obstacle_costs(ctrajectories, state, cost_params, time_points,
                                                                    predictor, reference_route)
    total_costs = np.sum(pointwise_costs, axis=1)  # costs per trajectory

    assert total_costs[0] < total_costs[1]              # obstacle-free route (smallest T_d)
    assert total_costs[1] < OBSTACLE_SIGMOID_COST       # close to obstacle route (medium T_d)
    assert total_costs[2] > OBSTACLE_SIGMOID_COST       # obstacle-colliding route (largest T_d)
