
from logging import Logger
from typing import List, Dict

import numpy as np
import time

from decision_making.src.global_constants import SAFETY_MARGIN_TIME_DELAY, SPECIFICATION_MARGIN_TIME_DELAY, \
    LON_ACC_LIMITS, SAFETY_SAMPLING_RESOLUTION, LAT_ACC_LIMITS, LATERAL_SAFETY_MU
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState, RelativeLane, \
    RelativeLongitudinalPosition, RoadSemanticOccupancyGrid
from decision_making.src.planning.behavioral.data_objects import ActionSpec, ActionRecipe, ActionType
from decision_making.src.planning.types import LIMIT_MIN, FS_SV, FS_SX, FrenetState2D, FS_SA, FS_DX, FS_DV, FS_DA, \
    FrenetTrajectory2D, FrenetTrajectories2D
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D
from mapping.src.service.map_service import MapService


class SafetyUtils:

    @staticmethod
    def calc_safety(behavioral_state: BehavioralGridState, ego_init_fstate: FrenetState2D,
                    recipes: List[ActionRecipe], specs: List[ActionSpec], action_specs_mask: List[bool],
                    predictions: Dict[int, FrenetTrajectory2D], time_samples: np.array) -> [np.array, np.array]:
        """
        Calculate longitudinally safe intervals w.r.t. F, LF/RF, LB/RB, that are permitted for for lateral motion
        for all action specifications.
        If an action does not perform lateral motion, it should be safe in the whole interval [0, spec.t].
        :param behavioral_state: the current behavioral state
        :param ego_init_fstate: the current ego Frenet state
        :param specs: list of action specifications
        :param action_specs_mask: array of boolean values: mask[i]=True if specs[i] was not filtered
        :return: for each action, array of time intervals inside [0, spec.t], where ego is safe
        """
        ego = behavioral_state.ego_state
        (ego_length, ego_width) = (ego.size.length, ego.size.width)
        ego_size = np.array([ego_length, ego_width])
        lane_width = MapService.get_instance().get_road(ego.road_localization.road_id).lane_width

        # create a matrix of all non-filtered specs with their original index
        specs_arr = np.array([np.array([i, spec.t, spec.v, spec.s, spec.d])
                              for i, spec in enumerate(specs) if action_specs_mask[i]])
        filtered_recipes = np.array([recipes[i] for i, spec in enumerate(specs) if action_specs_mask[i]])

        spec_orig_idxs = specs_arr[:, 0].astype(int)
        (specs_t, specs_v, specs_s, specs_d) = (specs_arr[:, 1], specs_arr[:, 2], specs_arr[:, 3], specs_arr[:, 4])

        # for each spec calculate the samples number until the real spec length: spec.t
        samples_num = time_samples.shape[0]
        actions_num = specs_t.shape[0]
        occupancy_grid = behavioral_state.road_occupancy_grid
        dynamic_objects = [occupancy_grid[cell][0].dynamic_object for cell in occupancy_grid]

        st = time.time()
        # calculate ego trajectories for all actions and time samples
        ego_sx, ego_sv = SafetyUtils._calc_longitudinal_ego_trajectories(ego_init_fstate, specs_t, specs_v, specs_s,
                                                                         time_samples)
        time1 = time.time()-st
        st = time.time()

        zeros = np.zeros(ego_sx.shape)
        ego_ftraj = np.dstack((ego_sx, ego_sv, zeros, zeros, zeros, zeros))

        obj_sizes = dict([(obj.obj_id, np.array([obj.size.length, obj.size.width]))
                          for obj in dynamic_objects if obj.obj_id in predictions])
        # find all pairs obj_spec_list = [(obj_id, spec_id)], for which the object is relevant to spec's safety
        # only for lane change actions
        lon_safe_times = SafetyUtils._calc_lon_safe_times(occupancy_grid, ego_init_fstate, ego_ftraj, ego_size,
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

        # time2 = time.time()-st
        # st = time.time()
        # # find the first time interval in lon_safe_times for each spec
        # # the first safe time
        # lon_safe_from = np.c_[lon_safe_times.astype(int), np.ones(actions_num)].argmax(axis=1)
        # # the first unsafe time, after lon_safe_from
        # lon_safe_till = [lon_safe_from[i] + np.append(row[lon_safe_from[i]:], 0).argmin()
        #                  for i, row in enumerate(lon_safe_times)]
        # T_d = np.minimum(specs_t, time_samples[np.clip(lon_safe_till, 0, samples_num - 1)])
        #
        # ego_dx, ego_dv = SafetyUtils._calc_lateral_ego_trajectories(ego_init_fstate, specs_t, specs_d,
        #                                                             T_d, time_samples)
        # ego_ftraj = np.dstack((ego_sx, ego_sv, zeros, ego_dx, ego_dv, zeros))
        # time4 = time.time()-st
        #
        # predictions_mat = np.array([pred for i, pred in predictions.items()])
        #
        # st = time.time()
        # safe_times = SafetyUtils.calc_safety_for_trajectories(ego_ftraj, ego_size, predictions_mat,
        #                                                       np.array(list(obj_sizes.values())))
        # safe_specs = np.repeat(None, len(specs))
        # T_d_full = np.zeros(len(specs))
        # T_d_full[spec_orig_idxs] = T_d
        #
        # safe_specs[spec_orig_idxs] = safe_times.all(axis=(1, 2))
        # time5 = time.time()-st
        print('time1=%f time2=%f time3=%f' % (time1, time2, time3))

        return intervals

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
    def _calc_longitudinal_ego_trajectories(ego_init_fstate: FrenetState2D,
                                            specs_t: np.array, specs_v: np.array, specs_s: np.array,
                                            time_samples: np.array) -> [np.array, np.array]:
        """
        Calculate longitudinal ego trajectory for the given time samples.
        :param ego_init_fstate:
        :param specs:
        :param time_samples:
        :return: x_profile, v_profile
        """
        # TODO: Acceleration is not calculated.

        # duplicate time_samples array actions_num times
        actions_num = specs_t.shape[0]
        dup_time_samples = np.repeat(time_samples, actions_num).reshape(len(time_samples), actions_num)

        ds = specs_s - ego_init_fstate[FS_SX]
        # profiles for the cases, when dynamic object is in front of ego
        x_t = QuinticPoly1D.distance_by_constraints(a_0=ego_init_fstate[FS_SA], v_0=ego_init_fstate[FS_SV],
                                                    v_T=specs_v, ds=ds, T=specs_t)
        sx = x_t(dup_time_samples)
        # set inf to samples outside specs_t
        outside_samples = np.where(dup_time_samples > specs_t)
        sx[outside_samples[0], outside_samples[1]] = np.inf
        sx = sx.transpose()

        v_t = QuinticPoly1D.velocity_by_constraints(a_0=ego_init_fstate[FS_SA], v_0=ego_init_fstate[FS_SV],
                                                    v_T=specs_v, ds=ds, T=specs_t)
        sv = np.transpose(v_t(dup_time_samples))
        return ego_init_fstate[FS_SX] + sx, sv

    @staticmethod
    def _calc_lateral_ego_trajectories(ego_init_fstate: FrenetState2D, specs_t: np.array, specs_d: np.array,
                                       T_d: np.array, time_samples: np.array) -> [np.array, np.array]:
        """
        Calculate lateral ego trajectory for the given time samples.
        :param ego_init_fstate:
        :param specs_t:
        :param specs_d:
        :param time_samples:
        :return: x_profile, v_profile
        """
        # TODO: Acceleration is not calculated.

        # duplicate time_samples array actions_num times
        actions_num = specs_t.shape[0]
        dup_time_samples = np.repeat(time_samples, actions_num).reshape(len(time_samples), actions_num)

        trans_times = np.transpose(dup_time_samples)
        dd = specs_d - ego_init_fstate[FS_DX]
        # profiles for the cases, when dynamic object is in front of ego
        d_t = QuinticPoly1D.distance_by_constraints(a_0=ego_init_fstate[FS_DA], v_0=ego_init_fstate[FS_DV],
                                                    v_T=0, ds=dd, T=T_d)
        dx = d_t(dup_time_samples)
        # set inf to samples outside specs_t
        outside_samples = np.where(dup_time_samples > specs_t)
        dx[outside_samples[0], outside_samples[1]] = np.inf
        dx = dx.transpose()

        v_t = QuinticPoly1D.velocity_by_constraints(a_0=ego_init_fstate[FS_DA], v_0=ego_init_fstate[FS_DV],
                                                    v_T=0, ds=dd, T=T_d)
        dv = np.transpose(v_t(dup_time_samples))
        return ego_init_fstate[FS_DX] + dx, dv

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

        # calculate longitudinal safety
        lon_safe_times = SafetyUtils.get_lon_safety(
            ego_ftraj_dup[..., FS_SX], ego_ftraj_dup[..., FS_SV], SAFETY_MARGIN_TIME_DELAY,
            obj_ftraj[..., FS_SX], obj_ftraj[..., FS_SV], SPECIFICATION_MARGIN_TIME_DELAY,
            0.5 * (ego_size[0] + obj_lengths))

        if not both_dimensions_flag:  # if only longitudinal safety
            return lon_safe_times

        # calculate lateral safety
        lat_safe_times = SafetyUtils.get_lat_safety(
            ego_ftraj_dup[..., FS_DX], ego_ftraj_dup[..., FS_DV], SAFETY_MARGIN_TIME_DELAY,
            obj_ftraj[..., FS_DX], obj_ftraj[..., FS_DV], SPECIFICATION_MARGIN_TIME_DELAY,
            0.5 * (ego_size[1] + obj_widths) + LATERAL_SAFETY_MU)

        return np.logical_or(lon_safe_times, lat_safe_times)

    @staticmethod
    def get_lon_safety(ego_lon: np.array, ego_vel: np.array, ego_time_delay: float,
                       obj_lon: np.array, obj_vel: np.array, obj_time_delay: float,
                       margins: np.array, max_brake: float=-LON_ACC_LIMITS[LIMIT_MIN]) -> np.array:
        """
        Calculate longitudinal safety between ego and another object for all timestamps.
        :param ego_lon: [m] object1 longitudes: tensor of shape: traj_num x objects_num x timestamps_num
        :param ego_vel: [m/s] object1 velocities: tensor of shape: traj_num x objects_num x timestamps_num
        :param ego_time_delay: [sec] object1 time delay
        :param obj_lon: [m] object2 longitudes: tensor of any shape that compatible with the shape of object1
        :param obj_vel: [m/s] object2 velocities: tensor of any shape that compatible with the shape of object1
        :param obj_time_delay: [sec] object2 time delay
        :param margins: [m] objects' lengths: matrix of size objects_num x timestamps_num
        :param max_brake: [m/s^2] maximal deceleration of both objects
        :return: [bool] longitudinal safety per timestamp. Tensor of the same shape as object1 or object2
        """
        dist = ego_lon - obj_lon
        sign = np.sign(dist)
        switch = 0.5 * (sign + 1)
        safe_dist = np.clip(np.divide(sign * (obj_vel ** 2 - ego_vel ** 2), 2 * max_brake), 0, None) + \
                    (1 - switch) * ego_vel * ego_time_delay + switch * obj_vel * obj_time_delay + margins
        return sign * dist > safe_dist

    @staticmethod
    def get_lat_safety(ego_pos: np.array, ego_vel: np.array, ego_time_delay: float,
                       obj_pos: np.array, obj_vel: np.array, obj_time_delay: float,
                       margins: np.array, max_brake: float=-LAT_ACC_LIMITS[LIMIT_MIN]) -> np.array:
        """
        Calculate lateral safety between ego and another object for all timestamps.
        :param ego_pos: [m] object1 longitudes: tensor of any shape
        :param ego_vel: [m/s] object1 velocities: tensor of any shape
        :param ego_time_delay: [sec] object1 time delay
        :param obj_pos: [m] object2 longitudes: tensor of any shape that compatible with the shape of object1
        :param obj_vel: [m/s] object2 velocities: tensor of any shape that compatible with the shape of object1
        :param obj_time_delay: [sec] object2 time delay
        :param margins: [m] objects' widths + mu: matrix of size objects_num x timestamps_num
        :param max_brake: [m/s^2] maximal deceleration of both objects
        :return: [bool] lateral safety per timestamp. Tensor of the same shape as object1 or object2
        """
        dist = ego_pos - obj_pos
        sign = np.sign(dist)
        safe_dist = np.clip(np.divide(sign * (obj_vel * np.abs(obj_vel) - ego_vel * np.abs(obj_vel)), 2 * max_brake),
                            0, None) + \
                    np.clip(-sign * ego_vel, 0, None) * ego_time_delay + \
                    np.clip(sign * obj_vel, 0, None) * obj_time_delay + \
                    margins
        return sign * dist > safe_dist
