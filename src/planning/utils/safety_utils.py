from typing import List, Dict

import numpy as np
import time

from decision_making.src.global_constants import SAFETY_MARGIN_TIME_DELAY, SPECIFICATION_MARGIN_TIME_DELAY, \
    LON_ACC_LIMITS, LAT_ACC_LIMITS, LATERAL_SAFETY_MU
from decision_making.src.planning.behavioral.behavioral_grid_state import RelativeLongitudinalPosition, RelativeLane, \
    RoadSemanticOccupancyGrid, BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionRecipe, ActionType, ActionSpec
from decision_making.src.planning.types import LIMIT_MIN, FrenetTrajectories2D, FrenetState2D, FS_DX, FS_DV, FS_DA, \
    FS_SX, FrenetTrajectory2D
from decision_making.src.planning.utils.optimal_control.generate_traj import TrajectoriesGenerator
from mapping.src.service.map_service import MapService


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
        lane_width = MapService.get_instance().get_road(ego.road_localization.road_id).lane_width

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

        obj_sizes = dict([(obj.obj_id, np.array([obj.size.length, obj.size.width]))
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
                             ego_ftraj: FrenetTrajectories2D, ego_size: np.array, obj_sizes: np.array,
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
                                occupancy_grid[(origin_lanes[spec_i], lon_same)][0].fstate[FS_SX] > ego_lon:
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
            lon_safe_times_per_obj[specs, obj_i] = SafetyUtils.calc_safety_for_trajectories(
                ego_ftraj[specs], ego_size, pred, obj_sizes[obj_id], False)

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
    def calc_safe_T_d(ego_fstate: FrenetState2D, ego_size: np.array, specs: np.array, specs_mask: np.array,
                      time_samples: np.array, predictions: np.array, obj_sizes: np.array) -> np.array:
        """
        Calculate maximal safe T_d for each given action spec.
        If an action is not safe, the appropriate T_d is 0.
        :param ego_fstate: initial ego frenet state
        :param ego_size: array of size 2
        :param specs: array of ActionSpec
        :param specs_mask: boolean array of size specs.shape[0]
        :param time_samples: array of time samples for the given predictions; samples_num = predictions.shape[1]
        :param predictions: 2D array of objects predictions; shape: objects_num x samples_num
        :param obj_sizes: 2D array of objects' sizes; shape: objects_num x 2
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
        grid_resolution = 1  # seconds between two adjacent T_d samples
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
        safe_times = SafetyUtils.calc_safety_for_trajectories(ego_ftraj, ego_size, predictions, obj_sizes)

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
    def calc_safety_for_trajectories(ego_ftraj: FrenetTrajectories2D, ego_size: np.array,
                                     obj_ftraj: np.array, obj_sizes: np.array,
                                     both_dimensions_flag: bool=True) -> np.array:
        """
        Calculate safety boolean tensor for different ego Frenet trajectories and objects' Frenet trajectories.
        :param ego_ftraj: ego Frenet trajectories: tensor of shape: traj_num x timestamps_num x Frenet state size
        :param ego_size: array of size 2: ego length, ego width
        :param obj_ftraj: one or array of objects Frenet trajectories: tensor of shape: objects_num x timestamps_num x Frenet state size
        :param obj_sizes: one or array of arrays of size 2: i-th row is i-th object's size
        :param both_dimensions_flag: if False then only longitudinal dimension is considered
        :return: [bool] safety per [ego trajectory, object, timestamp]. Tensor of shape: traj_num x objects_num x timestamps_num
        """
        (ego_traj_num, times_num, fstate_size) = ego_ftraj.shape
        if obj_ftraj.ndim > 2:  # multiple objects
            objects_num = obj_ftraj.shape[0]
            # duplicate ego_ftraj to the following dimensions: ego_traj_num, objects_num, timestamps_num, fstate (6)
            ego_ftraj_dup = np.tile(ego_ftraj, objects_num).reshape(ego_traj_num, times_num, objects_num,
                                                                    fstate_size).swapaxes(1, 2)
            obj_lengths = np.repeat(obj_sizes[:, 0], times_num).reshape(objects_num, times_num)
            obj_widths = np.repeat(obj_sizes[:, 1], times_num).reshape(objects_num, times_num)
        else:  # a single object, don't duplicate ego_ftraj and obj_sizes
            ego_ftraj_dup = ego_ftraj
            obj_lengths = obj_sizes[0]
            obj_widths = obj_sizes[1]

        # split the trajectories to 6 fstate components for the duplicated ego and the object
        ego = np.array(np.split(ego_ftraj_dup, 6, axis=-1))[..., 0]
        obj = np.array(np.split(obj_ftraj, 6, axis=-1))[..., 0]

        # calculate longitudinal safety
        lon_safe_times = SafetyUtils.get_lon_safety(ego, SAFETY_MARGIN_TIME_DELAY,
                                                         obj, SPECIFICATION_MARGIN_TIME_DELAY,
                                                         0.5 * (ego_size[0] + obj_lengths))

        if not both_dimensions_flag:  # if only longitudinal safety
            return lon_safe_times

        # calculate lateral safety
        lat_safe_times = SafetyUtils.get_lat_safety(ego, SAFETY_MARGIN_TIME_DELAY,
                                                    obj, SPECIFICATION_MARGIN_TIME_DELAY,
                                                    0.5 * (ego_size[1] + obj_widths) + LATERAL_SAFETY_MU)

        return np.logical_or(lon_safe_times, lat_safe_times)

    @staticmethod
    def get_lon_safety(ego: np.array, ego_time_delay: float, obj: np.array, obj_time_delay: float,
                       margins: np.array, max_brake: float=-LON_ACC_LIMITS[LIMIT_MIN]) -> np.array:
        """
        Calculate longitudinal safety between ego and another object for all timestamps.
        :param ego: ego fstate components: tensor of shape: 6 x traj_num x objects_num x timestamps_num
        :param ego_time_delay: [sec] ego time delay
        :param obj: object's fstate components: tensor of any shape that compatible with the shape of ego
        :param obj_time_delay: [sec] object's time delay
        :param margins: [m] lengths half sum: matrix of size objects_num x timestamps_num
        :param max_brake: [m/s^2] maximal deceleration of both objects
        :return: [bool] longitudinal safety per timestamp. Tensor of the same shape as object1 or object2
        """
        (ego_lon, ego_vel, _, _, _, _) = ego
        (obj_lon, obj_vel, _, _, _, _) = obj

        dist = ego_lon - obj_lon
        sign = np.sign(dist)
        ego_ahead = 0.5 * (sign + 1)
        safe_dist = np.clip(np.divide(sign * (obj_vel ** 2 - ego_vel ** 2), 2 * max_brake), 0, None) + \
                    (1 - ego_ahead) * ego_vel * ego_time_delay + ego_ahead * obj_vel * obj_time_delay + margins
        return sign * dist > safe_dist

    @staticmethod
    def get_lat_safety(ego: np.array, ego_time_delay: float, obj: np.array, obj_time_delay: float,
                       margins: np.array, max_brake: float=-LAT_ACC_LIMITS[LIMIT_MIN]) -> np.array:
        """
        Calculate lateral safety between ego and another object for all timestamps.
        :param ego: ego fstate components: tensor of shape: 6 x traj_num x objects_num x timestamps_num
        :param ego_time_delay: [sec] object1 time delay
        :param obj: object's fstate components: tensor of any shape that compatible with the shape of ego
        :param obj_time_delay: [sec] object2 time delay
        :param margins: [m] objects' widths + mu: matrix of size objects_num x timestamps_num
        :param max_brake: [m/s^2] maximal deceleration of both objects
        :return: [bool] lateral safety per timestamp. Tensor of the same shape as object1 or object2
        """
        (_, _, _, ego_lat, ego_lat_vel, _) = ego
        (_, _, _, obj_lat, obj_lat_vel, _) = obj

        dist = ego_lat - obj_lat
        sign = np.sign(dist)
        safe_dist = np.clip(np.divide(sign * (obj_lat_vel * np.abs(obj_lat_vel) - ego_lat_vel * np.abs(ego_lat_vel)),
                                      2 * max_brake),
                            0, None) + \
                    np.clip(-sign * ego_lat_vel, 0, None) * ego_time_delay + \
                    np.clip( sign * obj_lat_vel, 0, None) * obj_time_delay + \
                    margins
        return sign * dist > safe_dist
