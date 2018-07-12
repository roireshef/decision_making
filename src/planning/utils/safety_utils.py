from typing import List, Dict

import numpy as np
import time

from decision_making.src.global_constants import SAFETY_MARGIN_TIME_DELAY, SPECIFICATION_MARGIN_TIME_DELAY, \
    LON_ACC_LIMITS, LAT_ACC_LIMITS, LATERAL_SAFETY_MU, LON_SAFETY_ACCEL_DURING_DELAY, LAT_SAFETY_ACCEL_DURING_DELAY, \
    LAT_VEL_BLAME_THRESH
from decision_making.src.planning.behavioral.behavioral_grid_state import RelativeLongitudinalPosition, RelativeLane, \
    RoadSemanticOccupancyGrid, BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionRecipe, ActionType, ActionSpec
from decision_making.src.planning.types import LIMIT_MIN, FrenetTrajectories2D, FrenetState2D, FS_DX, FS_DV, FS_DA, \
    FS_SX, FrenetTrajectory2D, FS_SV
from decision_making.src.planning.utils.optimal_control.generate_traj import TrajectoriesGenerator
from decision_making.src.state.state import ObjectSize
from mapping.src.service.map_service import MapService


EGO_ACCEL_DIST = 0.5 * LON_SAFETY_ACCEL_DURING_DELAY * SAFETY_MARGIN_TIME_DELAY * SAFETY_MARGIN_TIME_DELAY
OBJ_ACCEL_DIST = 0.5 * LON_SAFETY_ACCEL_DURING_DELAY * SPECIFICATION_MARGIN_TIME_DELAY * SPECIFICATION_MARGIN_TIME_DELAY

class SafetyUtils:

    @staticmethod
    def calc_safe_intervals(behavioral_state: BehavioralGridState, ego_fstate: FrenetState2D,
                            recipes: List[ActionRecipe], specs: List[ActionSpec], specs_mask: List[bool],
                            predictions: Dict[int, FrenetTrajectory2D], time_samples: np.array) -> np.array:
        """
        Calculate longitudinally safe intervals w.r.t. F, LF/RF, LB/RB, that are permitted for for lateral motion
        for all action specifications.
        If an action does not perform lateral motion, it should be safe in the whole interval [0, spec.t].
        :return: for each action, array of time intervals inside [0, spec.t], where ego is safe
        """
        ego = behavioral_state.ego_state
        (ego_length, ego_width) = (ego.size.length, ego.size.width)
        ego_size = np.array([ego_length, ego_width])
        lane_width = MapService.get_instance().get_road(ego.map_state.road_id).lane_width

        # collect non-filtered specs details: t,v,s,d, and mapping between valid specs and all specs.
        specs_arr = np.array([np.array([i, spec.t, spec.v, spec.s, spec.d])
                              for i, spec in enumerate(specs) if specs_mask[i]])
        (spec_orig_idxs, specs_t, specs_v, specs_s, specs_d) = specs_arr.transpose()
        spec_orig_idxs = spec_orig_idxs.astype(int)
        filtered_recipes = np.array([recipes[i] for i, spec in enumerate(specs) if specs_mask[i]])

        # for each spec calculate the samples number until the real spec length: spec.t
        actions_num = specs_t.shape[0]
        samples_num = time_samples.shape[0]
        occupancy_grid = behavioral_state.road_occupancy_grid
        dynamic_objects = [occupancy_grid[cell][0].dynamic_object for cell in occupancy_grid]

        st_tot = time.time()
        st = time.time()
        # calculate ego trajectories for all actions and time samples
        ego_sx, ego_sv = TrajectoriesGenerator.calc_longitudinal_trajectories(ego_fstate, specs_t, specs_v, specs_s,
                                                                              time_samples)
        time1 = time.time()-st
        st = time.time()

        zeros = np.zeros(ego_sx.shape)
        ego_ftraj = np.dstack((ego_sx, ego_sv, zeros, zeros, zeros, zeros))

        obj_sizes = dict([(obj.obj_id, ObjectSize(obj.size.length, obj.size.width, 0))
                          for obj in dynamic_objects if obj.obj_id in predictions])
        # find all pairs obj_spec_list = [(obj_id, spec_id)], for which the object is relevant to spec's safety
        # only for lane change actions
        lon_safe_times = SafetyUtils._calc_lon_safe_times(occupancy_grid, ego_fstate, ego_ftraj, ego_size,
                                                          obj_sizes, filtered_recipes, specs_d, predictions,
                                                          samples_num, lane_width)
        time2 = time.time()-st
        st = time.time()

        # find safe intervals in lon_safe_times
        ext_safe_times = np.c_[np.repeat(False, actions_num), lon_safe_times, np.repeat(False, actions_num)]
        start_points = np.where(np.logical_and(ext_safe_times[:, 1:], np.logical_not(ext_safe_times[:, :-1])))
        end_points = np.where(np.logical_and(ext_safe_times[:, :-1], np.logical_not(ext_safe_times[:, 1:])))
        (start_times, end_times) = (time_samples[start_points[1]], time_samples[end_points[1]-1])
        intervals = np.array([(spec_orig_idxs[spec_i], start_times[i], min(end_times[i], specs_t[spec_i]))
                              for i, spec_i in enumerate(start_points[0]) if start_times[i] < specs_t[spec_i]])

        time3 = time.time()-st
        time_tot = time.time() - st_tot

        print('time_tot=%f: time1=%f time2=%f time3=%f' % (time_tot, time1, time2, time3))

        return intervals

    @staticmethod
    def _calc_lon_safe_times(occupancy_grid: RoadSemanticOccupancyGrid, ego_init_fstate: np.array,
                             ego_ftraj: FrenetTrajectories2D, ego_size: np.array, obj_sizes: Dict[int, ObjectSize],
                             recipes: np.array, specs_d: np.array, predictions: Dict[int, FrenetTrajectory2D],
                             samples_num, lane_width: float) -> np.array:
        """
        :param occupancy_grid:
        :param ego_init_fstate:
        :param ego_ftraj:
        :param ego_size:
        :param specs_d:
        :param lane_width:
        :return:
        """
        # get origin & target relative lane for all actions, based on the current and target latitudes
        # origin_lane may be different from the current ego lane, if ego elapsed > half lateral distance to target
        origin_lanes, target_lanes = SafetyUtils._get_rel_lane_from_specs(lane_width, ego_init_fstate, specs_d)
        ego_lon = ego_init_fstate[FS_SX]
        actions_num = specs_d.shape[0]
        ego_size_class = ObjectSize(ego_size[0], ego_size[1], 0)

        (lon_front, lon_same, lon_rear) = (RelativeLongitudinalPosition.FRONT, RelativeLongitudinalPosition.PARALLEL,
                                           RelativeLongitudinalPosition.REAR)
        # find all pairs obj_spec_list = [(obj_id, spec_id)], for which the object is relevant to spec's safety
        # only for lane change actions
        obj_spec_list = []
        # target_lane is is per spec (array of size actions_num)
        for spec_i, tar_lane in enumerate(target_lanes):
            recipe: ActionRecipe = recipes[spec_i]
            rel_lon = lon_front.value
            if recipe.action_type == ActionType.FOLLOW_VEHICLE:
                rel_lon = recipe.relative_lon.value
            elif recipe.action_type == ActionType.OVERTAKE_VEHICLE:
                rel_lon = recipe.relative_lon.value + 1

            # add followed object to obj_spec_list
            if rel_lon == lon_front.value:
                if (tar_lane, lon_front) in occupancy_grid:
                    obj_spec_list.append((occupancy_grid[(tar_lane, lon_front)][0].dynamic_object.obj_id, spec_i, 0))
            elif rel_lon == lon_same.value:
                if (tar_lane, lon_same) in occupancy_grid:
                    obj_spec_list.append((occupancy_grid[(tar_lane, lon_same)][0].dynamic_object.obj_id, spec_i, 0))
            elif rel_lon == lon_rear.value:
                if (tar_lane, lon_rear) in occupancy_grid:
                    obj_spec_list.append((occupancy_grid[(tar_lane, lon_rear)][0].dynamic_object.obj_id, spec_i, 0))
            elif rel_lon > lon_front.value:  # the followed object is ahead of lon_front
                if (tar_lane, lon_front) in occupancy_grid and len(occupancy_grid[(tar_lane, lon_front)]) > 1:
                    obj_spec_list.append((occupancy_grid[(tar_lane, lon_front)][1].dynamic_object.obj_id, spec_i, 0))

            if tar_lane != origin_lanes[spec_i]:  # lane change
                # add back object to obj_spec_list
                if rel_lon == lon_front.value:
                    if (tar_lane, lon_same) in occupancy_grid:
                        obj_spec_list.append((occupancy_grid[(tar_lane, lon_same)][0].dynamic_object.obj_id, spec_i, 2))
                    elif (tar_lane, lon_rear) in occupancy_grid:
                        obj_spec_list.append((occupancy_grid[(tar_lane, lon_rear)][0].dynamic_object.obj_id, spec_i, 2))
                elif rel_lon == lon_same.value:
                    if (tar_lane, lon_rear) in occupancy_grid:
                        obj_spec_list.append((occupancy_grid[(tar_lane, lon_rear)][0].dynamic_object.obj_id, spec_i, 2))
                elif rel_lon == lon_rear.value:  # the back object is the object behind lon_rear
                    if (tar_lane, lon_rear) in occupancy_grid and len(occupancy_grid[(tar_lane, lon_rear)]) > 1:
                        obj_spec_list.append((occupancy_grid[(tar_lane, lon_rear)][1].dynamic_object.obj_id, spec_i, 2))

                # add front object to obj_spec_list
                # the front object is in the original lane; it may be in lon_same or in lon_front
                if (origin_lanes[spec_i], lon_same) in occupancy_grid and \
                    occupancy_grid[(origin_lanes[spec_i], lon_same)][0].dynamic_object.map_state.road_fstate[FS_SX] > ego_lon:
                    obj_spec_list.append((occupancy_grid[(origin_lanes[spec_i], lon_same)][0].dynamic_object.obj_id,
                                          spec_i, 1))
                elif (origin_lanes[spec_i], lon_front) in occupancy_grid:
                    obj_spec_list.append((occupancy_grid[(origin_lanes[spec_i], lon_front)][0].dynamic_object.obj_id,
                                          spec_i, 1))

        obj_spec_list = np.array(obj_spec_list)
        obj_set = set(obj_spec_list[:, 0])

        lon_safe_times_per_obj = np.ones((actions_num, len(obj_set), samples_num)).astype(bool)
        obj_id_to_obj_i = {}
        for obj_i, obj_id in enumerate(obj_set):
            obj_id_to_obj_i[obj_id] = obj_i
            specs = obj_spec_list[np.where(obj_spec_list[:, 0] == obj_id)][:, 1]
            pred = predictions[obj_id]
            lon_safe_times_per_obj[specs, obj_i] = SafetyUtils.get_safe_times(
                ego_ftraj[specs], ego_size_class, pred[np.newaxis], [obj_sizes[obj_id]], False)[:, 0, :]

        # for each spec pick its follow and front objects, find the unsafe regions:
        # (before the last unsafe time w.r.t. follow object and after the first unsafe time w.r.t. front object)
        # and fill them by False
        for spec_i, spec_safe_times in enumerate(lon_safe_times_per_obj):
            follow_objects = obj_spec_list[np.where(np.logical_and(obj_spec_list[:, 1] == spec_i,
                                                                   obj_spec_list[:, 2] == 0))][:, 0]
            if follow_objects.shape[0] > 0:
                obj_i = obj_id_to_obj_i[follow_objects[0]]
                safe_times = lon_safe_times_per_obj[spec_i, obj_i]
                safe_from = samples_num - np.append(safe_times[::-1], False).argmin()  # last unsafe index w.r.t. LF
                safe_times[:safe_from] = False  # fill False before safe_from

            front_objects = obj_spec_list[np.where(np.logical_and(obj_spec_list[:, 1] == spec_i,
                                                                  obj_spec_list[:, 2] == 1))][:, 0]
            if front_objects.shape[0] > 0:
                obj_i = obj_id_to_obj_i[front_objects[0]]
                safe_times = lon_safe_times_per_obj[spec_i, obj_i]
                safe_till = np.append(safe_times, False).argmin()  # first unsafe index w.r.t. F
                safe_times[safe_till:] = False  # fill False after safe_till

        lon_safe_times = lon_safe_times_per_obj.all(axis=1)
        return lon_safe_times

    @staticmethod
    def _get_rel_lane_from_specs(lane_width: float, ego_fstate: FrenetState2D, specs_d: np.array) -> \
            [List[RelativeLane], List[RelativeLane]]:
        """
        Return origin and target relative lanes
        :param lane_width:
        :param ego_fstate:
        :param specs_d:
        :return: origin & target relative lanes
        """
        specs_num = specs_d.shape[0]
        d_dist = specs_d - ego_fstate[FS_DX]

        # same_same_idxs = np.where(-lane_width/4 <= d_dist <= lane_width/4)[0]
        # right_same_idxs = np.where(np.logical_and(lane_width / 4 < d_dist, d_dist <= lane_width / 2))
        # left_same_idxs = np.where(np.logical_and(-lane_width / 2 <= d_dist, d_dist < -lane_width / 4))
        same_left_idxs = np.where(d_dist > lane_width / 2)
        same_right_idxs = np.where(d_dist < -lane_width / 2)

        origin_lanes = np.array([RelativeLane.SAME_LANE] * specs_num)
        target_lanes = np.array([RelativeLane.SAME_LANE] * specs_num)
        # origin_lanes[right_same_idxs] = RelativeLane.RIGHT_LANE
        # origin_lanes[left_same_idxs] = RelativeLane.LEFT_LANE
        target_lanes[same_left_idxs] = RelativeLane.LEFT_LANE
        target_lanes[same_right_idxs] = RelativeLane.RIGHT_LANE
        return list(origin_lanes), list(target_lanes)

    @staticmethod
    def calc_safe_T_d(ego_fstate: FrenetState2D, ego_size: ObjectSize, specs: np.array, specs_mask: np.array,
                      time_samples: np.array, predictions: np.array, obj_sizes: List[ObjectSize]) -> np.array:
        """
        Calculate maximal safe T_d for each given action spec.
        If an action is not safe, the appropriate T_d is 0.
        :param ego_fstate: initial ego frenet state
        :param ego_size: ego size
        :param specs: array of ActionSpec
        :param specs_mask: boolean array of size specs.shape[0]
        :param time_samples: array of time samples for the given predictions; samples_num = predictions.shape[1]
        :param predictions: 2D array of objects predictions; shape: objects_num x samples_num
        :param obj_sizes: list of objects' sizes
        :return: 1D array of T_d per spec.
        """
        samples_num = time_samples.shape[0]
        # collect non-filtered specs details: t,v,s,d, and mapping between valid specs and all specs.
        specs_arr = np.array([np.array([i, spec.t, spec.v, spec.s, spec.d])
                              for i, spec in enumerate(specs) if specs_mask[i]])
        (spec_orig_idxs, specs_t, specs_v, specs_s, specs_d) = specs_arr.transpose()
        spec_orig_idxs = spec_orig_idxs.astype(int)

        st_tot = time.time()
        st = time.time()
        valid_specs_num = specs_t.shape[0]  # after filtering

        # choose a grid of T_d between reasonable bounds
        grid_resolution = 2  # seconds between two adjacent T_d samples
        T_d = np.arange(7, 3 - np.finfo(np.float16).eps, -grid_resolution)
        T_d_num = T_d.shape[0]

        # calculate ego lateral trajectories (only location and velocity) for all specs and all T_d values
        ego_dx, ego_dv = TrajectoriesGenerator.calc_lateral_trajectories(ego_fstate, specs_d, T_d, time_samples)
        lat_time = time.time()-st
        st = time.time()

        # calculate ego longitudinal trajectories (only location and velocity) for all specs
        ego_sx, ego_sv = TrajectoriesGenerator.calc_longitudinal_trajectories(ego_fstate, specs_t, specs_v, specs_s,
                                                                              time_samples)
        lon_time = time.time()-st
        st = time.time()
        # duplicate longitudinal trajectories to be aligned with lateral trajectories
        dup_ego_sx = np.tile(ego_sx, T_d_num).reshape(specs_t.shape[0] * T_d_num, samples_num)
        dup_ego_sv = np.tile(ego_sv, T_d_num).reshape(specs_t.shape[0] * T_d_num, samples_num)
        zeros = np.zeros(dup_ego_sx.shape)
        # ego Frenet trajectories
        ego_ftraj = np.dstack((dup_ego_sx, dup_ego_sv, zeros, ego_dx, ego_dv, zeros))

        dup_time = time.time()-st
        st = time.time()

        # calculate RSS for all trajectories and all time samples
        # returns 3D array; safe times for all valid specs, T_d's, objects, time samples;
        # shape: valid_specs_num*T_d_num x objects_num x samples_num
        safe_times = SafetyUtils.get_safe_times(ego_ftraj, ego_size, predictions, obj_sizes)

        RSS_time = time.time()-st

        # 2D array; safe times for all actions and T_d's; shape: valid_specs x T_d_num
        safe_specs_T_d = safe_times.all(axis=(1, 2)).reshape(valid_specs_num, T_d_num)

        # 1D array; unsafe actions; shape: actions_num
        unsafe_specs = np.logical_not(safe_specs_T_d).all(axis=1)

        # 1D array; maximal safe T_d for each valid spec; shape: valid_specs
        max_safe_T_d = T_d[np.argmax(safe_specs_T_d, axis=1)]  # for each spec find max T_d
        max_safe_T_d[unsafe_specs] = 0
        max_safe_T_d = np.minimum(max_safe_T_d, specs_t)

        # 1D array; maximal safe T_d for each spec; shape: specs.shape[0]
        safe_T_d = np.zeros(len(specs))
        safe_T_d[spec_orig_idxs[np.arange(0, valid_specs_num)]] = max_safe_T_d

        time_tot = time.time() - st_tot

        print('calc_safe_T_d time=%f: lat=%f lon=%f dup=%f RSS=%f' % (time_tot, lat_time, lon_time, dup_time, RSS_time))
        return safe_T_d

    @staticmethod
    def get_safe_times(ego: FrenetTrajectories2D, ego_size: ObjectSize, obj: np.array, obj_sizes: List[ObjectSize],
                       both_dimensions: bool=True) -> np.array:
        """
        Calculate safety boolean tensor for different ego Frenet trajectories and objects' Frenet trajectories.
        :param ego: ego Frenet trajectories: 3D tensor of shape: traj_num x timestamps_num x Frenet state size
        :param ego_size: ego size
        :param obj: 3D array of ftrajectories of objects: shape: objects_num x timestamps_num x 6 (Frenet state size)
        :param obj_sizes: list of objects' sizes
        :return: [bool] safety per [ego trajectory, object, timestamp]. Tensor of shape: traj_num x objects_num x timestamps_num
        """
        objects_num = obj.shape[0]
        # calculate blame times for every object
        safe_times = np.ones((ego.shape[0], objects_num, ego.shape[1])).astype(bool)
        ego_size_arr = np.array([ego_size.length, ego_size.width])
        for i in range(objects_num):  # loop over objects
            obj_size_arr = np.array([obj_sizes[i].length, obj_sizes[i].width])
            safe_times[:, i] = SafetyUtils._get_safe_times_per_obj(ego, ego_size_arr, obj[i], obj_size_arr,
                                                                   both_dimensions)
        return safe_times

    @staticmethod
    def _get_safe_times_per_obj(ego: np.array, ego_size: np.array, obj: np.array, obj_size: np.array,
                                both_dimensions: bool=True) -> np.array:
        """
        Calculate safety boolean tensor for different ego Frenet trajectories and a single object.
        :param ego: ego frenet trajectories: 3D tensor of shape: traj_num x timestamps_num x 6
        :param ego_size: array of size 2: ego length, ego width
        :param obj: object's frenet trajectories: 2D matrix of shape timestamps_num x 6
        :param obj_size: one or array of arrays of size 2: i-th row is i-th object's size
        :return: [bool] ego blame per [ego trajectory, timestamp]. 2D matrix of shape: traj_num x timestamps_num
        """
        lon_margin = 0.5 * (ego_size[0] + obj_size[0])
        lat_margin = 0.5 * (ego_size[1] + obj_size[1]) + LATERAL_SAFETY_MU

        # calculate longitudinal safety
        lon_safe_times = SafetyUtils._get_lon_safety(ego, SAFETY_MARGIN_TIME_DELAY, obj, SPECIFICATION_MARGIN_TIME_DELAY,
                                                     lon_margin)
        if not both_dimensions:
            return lon_safe_times

        # calculate lateral safety
        lat_safe_times, lat_vel_blame = SafetyUtils._get_lat_safety(ego, SAFETY_MARGIN_TIME_DELAY, obj,
                                                                    SPECIFICATION_MARGIN_TIME_DELAY, lat_margin)
        # calculate and return blame times
        safe_times = SafetyUtils._calc_fully_safe_times(ego[..., FS_SX], obj[..., FS_SX], lon_safe_times,
                                                        lat_safe_times, lat_vel_blame, lon_margin)
        return safe_times

    @staticmethod
    def _get_lon_safety(ego: np.array, ego_response_time: float, obj: np.array, obj_response_time: float,
                        margin: float, max_brake: float=-LON_ACC_LIMITS[LIMIT_MIN]) -> np.array:
        """
        Calculate longitudinal safety between ego and another object for all timestamps.
        :param ego: ego frenet trajectories: 3D tensor of shape: traj_num x timestamps_num x 6
        :param ego_response_time: [sec] ego response time
        :param obj: object's frenet trajectories: 2D matrix: timestamps_num x 6
        :param obj_response_time: [sec] object's response time
        :param margin: [m] cars' lengths half sum
        :param max_brake: [m/s^2] maximal deceleration of both objects
        :return: [bool] longitudinal safety per timestamp. 2D matrix shape: traj_num x timestamps_num
        """
        ego_lon, ego_vel = ego[..., FS_SX], ego[..., FS_SV]
        obj_lon, obj_vel = obj[..., FS_SX], obj[..., FS_SV]

        dist = ego_lon - obj_lon
        sign = np.sign(dist)
        ego_ahead = (sign > 0).astype(int)
        delayed_ego_vel = ego_vel + (1-ego_ahead) * ego_response_time * LON_SAFETY_ACCEL_DURING_DELAY
        delayed_obj_vel = obj_vel + ego_ahead * obj_response_time * LON_SAFETY_ACCEL_DURING_DELAY

        safe_dist = np.maximum(np.divide(sign * (delayed_obj_vel ** 2 - delayed_ego_vel ** 2), 2 * max_brake), 0) + \
                    (1 - ego_ahead) * (ego_vel * ego_response_time + EGO_ACCEL_DIST) + \
                    ego_ahead * (obj_vel * obj_response_time + OBJ_ACCEL_DIST) + margin
        return sign * dist > safe_dist

    @staticmethod
    def _get_lat_safety(ego: np.array, ego_response_time: float, obj: np.array, obj_response_time: float,
                        margin: float, max_brake: float=-LAT_ACC_LIMITS[LIMIT_MIN]) -> np.array:
        """
        Calculate lateral safety between ego and another object for all timestamps.
        :param ego: ego frenet trajectories: 3D tensor of shape: traj_num x timestamps_num x 6
        :param ego_response_time: [sec] object1 response time
        :param obj: object's frenet trajectories: 2D matrix: timestamps_num x 6
        :param obj_response_time: [sec] object2 response time
        :param margin: [m] half sum of objects' widths + mu (mu is from Mobileye's paper)
        :param max_brake: [m/s^2] maximal deceleration of both objects
        :return: [bool] 1. lateral safety per timestamp. 2D matrix shape: traj_num x timestamps_num
                        2. lateral velocity blame: True if ego moves laterally towards object. The same shape.
        """
        ego_lat, ego_lat_vel = ego[..., FS_DX], ego[..., FS_DV]
        obj_lat, obj_lat_vel = obj[..., FS_DX], obj[..., FS_DV]

        dist = ego_lat - obj_lat
        sign = np.sign(dist)

        delayed_ego_vel = ego_lat_vel - sign * ego_response_time * LAT_SAFETY_ACCEL_DURING_DELAY
        delayed_obj_vel = obj_lat_vel + sign * obj_response_time * LAT_SAFETY_ACCEL_DURING_DELAY
        avg_ego_vel = 0.5 * (ego_lat_vel + delayed_ego_vel)
        avg_obj_vel = 0.5 * (obj_lat_vel + delayed_obj_vel)

        reaction_dist = sign * (avg_obj_vel * obj_response_time - avg_ego_vel * ego_response_time)

        safe_dist = np.maximum(np.divide(sign * (delayed_obj_vel * np.abs(delayed_obj_vel) -
                                                 delayed_ego_vel * np.abs(delayed_ego_vel)),
                                         2 * max_brake) + reaction_dist,
                               0) + \
                    margin

        # lateral velocity blame is true if one of the 3 following conditions holds:
        #   ego moves laterally towards object faster than small thresh or
        #   ego moves laterally towards object faster than the object moves laterally towards ego or
        #   the action is towards the object
        action_lat_dir = (ego_lat[:, -1] - ego_lat[:, 0])[:, np.newaxis]  # last lat - initial lat
        lat_vel_blame = np.logical_or(-sign * ego_lat_vel > np.clip(sign * obj_lat_vel, 0, LAT_VEL_BLAME_THRESH),
                                      -sign * action_lat_dir > 0)
        return sign * dist > safe_dist, lat_vel_blame

    @staticmethod
    def _calc_fully_safe_times(ego_lon: np.array, obj_lon: np.array, lon_safe_times: np.array, lat_safe_times: np.array,
                               lat_vel_blame: np.array, lon_margin: float) -> np.array:
        """
        For all trajectories and times check if ego is safe wrt non-rear objects and not unsafe wrt rear on ego blame.
        A timestamp is defined as safe if
            1. ego is safe wrt non-rear object AND
            2. ego is not blamed for unsafe situation (wrt rear object)
        :param ego_lon: ego longitudes: 2D matrix of shape: traj_num x timestamps_num
        :param obj_lon: object longitudes: 1D array of size timestamps_num
        :param lon_safe_times: longitudinally safe times; shape as ego_lon
        :param lat_safe_times: laterally safe times; shape as ego_lon
        :param lat_vel_blame: times for which ego lat_vel towards object is larger than object's lat_vel; shape as ego_lon
        :param lon_margin: [m] cars' lengths half sum
        :return: 2D boolean matrix (shape as ego_lon) of times, when ego is safe and not blamed.
        """
        # find points, for which longitudinal safety changes from true to false, while unsafe laterally,
        # and the object is not behind ego
        lon_blame = np.logical_and(lon_safe_times[:, :-1], np.logical_not(lon_safe_times[:, 1:]))  # become unsafe
        lon_blame = np.insert(lon_blame, 0, np.logical_not(lon_safe_times[:, 0]), axis=-1)  # add first column
        lon_blame = np.logical_and(lon_blame, np.logical_not(lat_safe_times))  # becomes fully unsafe
        lon_blame = np.logical_and(lon_blame, (ego_lon < obj_lon))  # don't blame for unsafe rear object

        # find points, for which lateral safety changes from true to false, while the unsafe longitudinally,
        # and lateral velocity of ego towards the object is larger than of the object towards ego
        lat_blame = np.logical_and(lat_safe_times[:, :-1], np.logical_not(lat_safe_times[:, 1:]))  # become unsafe
        # blame in time=0: laterally and longitudinally unsafe at time=0 AND ego is behind object
        init_blame = np.logical_and(np.logical_not(lat_safe_times[:, 0]), lon_blame[:, 0])
        lat_blame = np.insert(lat_blame, 0, init_blame, axis=-1)  # add first column
        lat_blame = np.logical_and(lat_blame, np.logical_not(lon_safe_times))  # becomes fully unsafe
        lat_blame = np.logical_and(lat_blame, lat_vel_blame)  # blame according to lateral velocity

        # calculate final safety: a timestamp is defined as safe if
        #   1. ego is safe wrt non-rear object AND
        #   2. ego is not blamed for unsafe situation (wrt rear object)
        not_blame = np.logical_not(np.logical_or(lon_blame, lat_blame))
        safe_or_rear_obj = np.logical_or(ego_lon > obj_lon + lon_margin, np.logical_or(lon_safe_times, lat_safe_times))
        safe_times = np.logical_and(safe_or_rear_obj, not_blame)
        return safe_times
