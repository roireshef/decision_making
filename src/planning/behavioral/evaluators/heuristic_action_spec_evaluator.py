from logging import Logger
from typing import List, Optional

import numpy as np
import sys

from decision_making.src.global_constants import SAFE_DIST_TIME_DELAY, AV_TIME_DELAY, AGGRESSIVENESS_TO_LAT_ACC, \
    MINIMAL_STATIC_ACTION_TIME
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState, \
    RelativeLongitudinalPosition
from decision_making.src.planning.behavioral.data_objects import ActionRecipe, ActionType, \
    AggressivenessLevel, RelativeLane, NavigationGoal, ActionSpec
from decision_making.src.planning.behavioral.evaluators.action_evaluator import ActionSpecEvaluator
from decision_making.src.planning.behavioral.evaluators.cost_functions import \
    BP_EfficiencyMetric, \
    BP_ComfortMetric, BP_RightLaneMetric, VelocityProfile, BP_LaneDeviationMetric
from decision_making.src.planning.types import FP_SX, FS_DV, FS_DX, FS_SX, FS_SV
from decision_making.src.planning.utils.map_utils import MapUtils
from decision_making.src.state.state import NewEgoState
from mapping.src.service.map_service import MapService


class HeuristicActionSpecEvaluator(ActionSpecEvaluator):
    """
    Link to the algorithm documentation in confluence:
    https://confluence.gm.com/display/SHAREGPDIT/BP+costs+and+heuristic+assumptions
    """
    def __init__(self, logger: Logger):
        super().__init__(logger)
        self.back_danger_lane = None
        self.back_danger_side = None
        self.back_danger_time = None
        self.front_blame = False

    def evaluate(self, behavioral_state: BehavioralGridState, action_recipes: List[ActionRecipe],
                 action_specs: List[ActionSpec], action_specs_mask: List[bool]) -> np.ndarray:
        """
        Gets a list of actions to evaluate and returns a vector representing their costs.
        A set of actions is provided, enabling us to assess them independently.
        Note: the semantic actions were generated using the behavioral state and don't necessarily capture
        all relevant details in the scene. Therefore the evaluation is done using the behavioral state.
        :param behavioral_state: semantic actions grid behavioral state
        :param action_recipes: array of actions recipes
        :param action_specs: array of action specs
        :param action_specs_mask: array of boolean values indicating whether the spec was filtered or not
        :return: array of costs (one cost per action)
        """
        ego: NewEgoState = behavioral_state.ego_state
        ego_fstate = ego.map_state.road_state
        ego_segment = MapService.get_instance().get_segment(ego.map_state.segment_id)
        ego_lane = ego.map_state.lane_id  # TODO: lane_id = lane index in the segment
        center_lane_latitudes = ego_segment.get_center_lanes_latitudes(ego.map_state.segment_state[FS_SX])
        lane_width = HeuristicActionSpecEvaluator._get_lane_width(ego_fstate, ego_lane, center_lane_latitudes)

        print('\ntime=%.1f ego_v=%.2f ego_lat=%.2f ego_dv=%.2f grid_size=%d' %
              (ego.timestamp_in_sec, ego_fstate[FS_SV], ego_fstate[FS_DX], ego_fstate[FS_DV], len(behavioral_state.road_occupancy_grid)))

        costs = np.full(len(action_recipes), np.inf)

        for i, spec in enumerate(action_specs):
            if not action_specs_mask[i]:
                continue

            recipe = action_recipes[i]
            target_lane = ego_lane + recipe.relative_lane.value
            lat_dist = center_lane_latitudes[target_lane] - ego_fstate[FS_DX]

            # calculate lateral time according to the CALM aggressiveness level
            T_d_calm = VelocityProfile.calc_lateral_time(ego_fstate[FS_DV], lat_dist, AggressivenessLevel.CALM)

            # create velocity profile, whose length is at least as the lateral time
            vel_profile = HeuristicActionSpecEvaluator._calc_velocity_profile(ego_fstate, recipe, spec)
            if vel_profile is None:
                continue  # infeasible action

            # calculate the latest safe time
            T_d_max = self._calc_largest_safe_time(behavioral_state, recipe, i, vel_profile, ego.size.length,
                                                   T_d_calm, lane_width)
            if T_d_max < 0:
                continue  # the action is unsafe from the beginning
            T_d_max = min(T_d_max, T_d_calm)

            sub_costs = HeuristicActionSpecEvaluator._calc_action_costs(ego_fstate, vel_profile, spec, target_lane,
                                                                        lane_width, T_d_max)
            costs[i] = np.sum(sub_costs)

            print('action %d(%d %d) lane %d: dist=%.1f [td=%.2f t=%.2f s=%.2f v=%.2f] [v_mid=%.2f a=%.2f] '
                  '[eff %.3f comf %.2f,%.2f right %.2f dev %.2f]: tot %.2f' %
                  (i, recipe.action_type.value, recipe.aggressiveness.value, ego_lane + recipe.relative_lane.value,
                   HeuristicActionSpecEvaluator._dist_to_target(behavioral_state, recipe),
                   T_d_calm, spec.t, spec.s - ego_fstate[0], spec.v, vel_profile.v_mid,
                   (vel_profile.v_mid-vel_profile.v_init)/vel_profile.t1,
                   sub_costs[0], sub_costs[1], sub_costs[2], sub_costs[3], sub_costs[4], costs[i]))

        if np.isinf(np.min(costs)):
            print('*************************************************************')
            print('********************  NO SAFE ACTION!  **********************')
            print('*************************************************************')
        else:
            best_action = int(np.argmin(costs))
            print('Best action %d; lane %d\n' % (best_action, ego_lane + action_recipes[best_action].relative_lane.value))
        return costs

    @staticmethod
    def _calc_velocity_profile(ego_fstate: np.array, recipe: ActionRecipe, spec: ActionSpec) -> VelocityProfile:
        """
        Given action recipe and behavioral state, calculate the longitudinal velocity profile for that action.
        :param ego_fstate: current ego Frenet state
        :param recipe: the input action
        :param spec: action specification
        :return: longitudinal velocity profile or None if the action is infeasible by given aggressiveness level
        """
        if recipe.action_type == ActionType.FOLLOW_VEHICLE or recipe.action_type == ActionType.OVER_TAKE_VEHICLE:
            dist = spec.s - spec.t * spec.v - ego_fstate[FS_SX]
            return VelocityProfile.calc_profile_given_T(ego_fstate[FS_SV], spec.t, dist, spec.v)
        else:  # static action (FOLLOW_LANE)
            t2 = max(0., MINIMAL_STATIC_ACTION_TIME - spec.t)  # TODO: remove this hack after implementation of value function
            return VelocityProfile(v_init=ego_fstate[FS_SV], t1=spec.t, v_mid=spec.v, t2=t2, t3=0, v_tar=spec.v)

    def _calc_largest_safe_time(self, behavioral_state: BehavioralGridState, action: ActionRecipe, i: int,
                                vel_profile: VelocityProfile, ego_length: float, T_d: float, lane_width: float) -> float:
        """
        For a lane change action, given ego velocity profile and behavioral_state, get two cars that may
        require faster lateral movement (the front overtaken car and the back interfered car) and calculate the last
        time, for which the safety holds w.r.t. these two cars.
        :param behavioral_state: semantic actions grid behavioral state
        :param vel_profile: the velocity profile of ego
        :param ego_length: half ego length
        :param T_d: time for comfortable lane change
        :param lane_width: lane width of the road
        :return: the latest time, when ego is still safe; return -1 if the current state is unsafe for this action
        """
        action_lat_cell = action.relative_lane
        ego: NewEgoState = behavioral_state.ego_state
        ego_fstate = ego.map_state.road_state  # TODO: change it to the navigation frame state
        ego_lane = ego.map_state.lane_id  # TODO: lane_id = lane index in the segment
        intra_lane_lat = ego.map_state.lane_state[FS_DX]
        ego_lon = ego_fstate[FS_SX]
        cur_time = behavioral_state.ego_state.timestamp_in_sec
        T = vel_profile.total_time()

        forward_cell = (action_lat_cell, RelativeLongitudinalPosition.FRONT)
        front_cell = (RelativeLane.SAME_LANE, RelativeLongitudinalPosition.FRONT)
        side_rear_cell = (action_lat_cell, RelativeLongitudinalPosition.REAR)
        rear_cell = (RelativeLane.SAME_LANE, RelativeLongitudinalPosition.REAR)

        lat_dist_to_target = abs(action_lat_cell.value - (intra_lane_lat / lane_width - 0.5))  # in [0, 1.5]
        additional_time_delay = 0
        if action_lat_cell != RelativeLane.SAME_LANE and ego_fstate[FS_DV] * action_lat_cell.value <= 0:
            additional_time_delay = 0.2  # increase time delay if ego moves laterally against the target

        big_follow_td = AV_TIME_DELAY + 0.8  # follow lane TODO: remove it after value function implementation
        min_front_td = 0.0  # dist to F after completing lane change. TODO: increase it when the planning will be deep

        # check safety w.r.t. the front object on the target lane (if exists)
        if forward_cell in behavioral_state.road_occupancy_grid:
            followed_obj = behavioral_state.road_occupancy_grid[forward_cell][0].dynamic_object
            obj_fstate = followed_obj.map_state.road_state  # TODO: change it to the navigation frame state
            obj_lon = obj_fstate[FS_SX]

            # TODO: temp hack: smaller delay in following front car, to enable calm distance increasing if dist < 2 sec
            # TODO: the hack should be removed after value function implementation. Then td = AV_TIME_DELAY
            # without the hack this action is unsafe and filtered, and ego must move to another lane
            if action.action_type == ActionType.FOLLOW_VEHICLE:
                td = AV_TIME_DELAY
            else:
                td = big_follow_td
            td += 2 * additional_time_delay

            forward_safe_time = vel_profile.calc_last_safe_time(
                ego_lon, ego_length, obj_lon, obj_fstate[FS_SV], followed_obj.size.length, np.inf, td, td)
            if forward_safe_time < T:
                print('forward unsafe: %d(%d %d) rel_lat=%d dist=%.2f t=%.2f final_dist=%.2f v_obj=%.2f '
                      'prof=(t=[%.2f %.2f %.2f] v=[%.2f %.2f %.2f]) safe_time=%.2f td=%.2f' %
                      (i, action.action_type.value, action.aggressiveness.value, action_lat_cell.value, obj_lon - ego_lon,
                       T, obj_lon + T * followed_obj.v_x - (ego_lon + vel_profile.total_dist()),
                       followed_obj.v_x, vel_profile.t1, vel_profile.t2, vel_profile.t3, vel_profile.v_init,
                       vel_profile.v_mid, vel_profile.v_tar, forward_safe_time, td))
                return -1

        safe_time = np.inf
        if action_lat_cell != RelativeLane.SAME_LANE:
            # TODO: move it to a filter
            # check whether there is a car in the neighbor cell (same longitude)
            if (action_lat_cell, RelativeLongitudinalPosition.PARALLEL) in behavioral_state.road_occupancy_grid:
                print('side unsafe rel_lat=%d: action %d' % (action_lat_cell.value, i))
                return -1

            # check safety w.r.t. the front object on the original lane (if exists)
            if front_cell in behavioral_state.road_occupancy_grid:
                front_obj = behavioral_state.road_occupancy_grid[front_cell][0].dynamic_object
                obj_fstate = front_obj.map_state.road_state  # TODO: change it to the navigation frame state
                # time delay decreases as function of lateral distance to the target: td_0 = td_T + 1
                # td_0 > td_T, since as latitude advances ego can escape laterally easier
                td_T = min_front_td + additional_time_delay
                td_0 = AV_TIME_DELAY * lat_dist_to_target + additional_time_delay
                front_safe_time = vel_profile.calc_last_safe_time(
                    ego_lon, ego_length, obj_fstate[FS_SX], obj_fstate[FS_SV], front_obj.size.length, 0.75 * T_d, td_0, td_T)
                if front_safe_time < np.inf:
                    print('front_safe_time=%.2f action %d(%d %d): front_dist=%.2f front_vel=%.2f lat_d=%.2f td_0=%.2f td_T=%.2f' %
                          (front_safe_time, i, action.action_type.value, action.aggressiveness.value,
                           obj_fstate[FS_SX] - ego_lon, obj_fstate[FS_SV], lat_dist_to_target, td_0, td_T))
                if front_safe_time <= 0:
                    return -1
                safe_time = min(safe_time, front_safe_time)

            # check safety w.r.t. the back object on the original lane (if exists)
            if side_rear_cell in behavioral_state.road_occupancy_grid:
                back_obj = behavioral_state.road_occupancy_grid[side_rear_cell][0].dynamic_object
                obj_fstate = back_obj.map_state.road_state  # TODO: change it to the navigation frame state

                td = SAFE_DIST_TIME_DELAY + 3*additional_time_delay
                back_safe_time = vel_profile.calc_last_safe_time(
                    ego_lon, ego_length, obj_fstate[FS_SX], obj_fstate[FS_SV], back_obj.size.length, T_d, td, td)
                if back_safe_time < np.inf:
                    print('back_safe_time=%.2f action %d(%d %d): back_dist=%.2f back_vel=%.2f rel_lat=%.2f td=%.2f' %
                          (back_safe_time, i, action.action_type.value, action.aggressiveness.value,
                           ego_lon - obj_fstate[FS_SX], obj_fstate[FS_SV], action_lat_cell.value, td))
                # if ego is unsafe w.r.t. back_obj, then save a flag for the case ego will enter to its lane,
                # such that ego will check safety w.r.t to the rear object R
                if back_safe_time <= 0 <= ego_fstate[FS_DV] * action_lat_cell.value:  # ego moves laterally to target
                    self.back_danger_lane = ego_lane + action_lat_cell.value
                    self.back_danger_side = action_lat_cell.value  # -1 or 1
                    self.back_danger_time = cur_time
                if back_safe_time <= 0:
                    return -1
                safe_time = min(safe_time, back_safe_time)

        # check safety w.r.t. the rear object R for the case we are after back danger and arrived to the dangerous lane
        if self.back_danger_lane is not None:
            if cur_time - self.back_danger_time < 4:  # the danger is still relevant
                # if ego is on the danger_lane but still didn't reach the lane center,
                # and if this action is to the danger_lane center, then check safety w.r.t. the rear object
                if self.back_danger_lane == ego_lane and self.back_danger_side == action_lat_cell.value and \
                                intra_lane_lat * action_lat_cell.value < 0 and \
                                rear_cell in behavioral_state.road_occupancy_grid:
                    td = SAFE_DIST_TIME_DELAY
                    rear_obj = behavioral_state.road_occupancy_grid[rear_cell][0].dynamic_object
                    obj_fstate = rear_obj.map_state.road_state  # TODO: change it to the navigation frame state
                    back_safe_time = vel_profile.calc_last_safe_time(
                        ego_lon, ego_length, obj_fstate[FS_SX], obj_fstate[FS_SV], rear_obj.size.length, T_d, td, td)
                    if back_safe_time <= 0:
                        return -1
                    safe_time = min(safe_time, back_safe_time)
            else:  # after timeout, delete the back_danger flag
                self.back_danger_lane = None

        # print('front_time=%f back_time=%f forward_time=%f safe_time=%f' % \
        # (front_safe_time, back_safe_time, forward_safe_time, safe_time))
        return safe_time

    @staticmethod
    def _calc_action_costs(ego_fstate: np.array, vel_profile: VelocityProfile, action_spec: ActionSpec,
                           target_lane: int, lane_width: float, T_d: float) -> [float, np.array]:
        """
        Calculate the cost of the action
        :param vel_profile: longitudinal velocity profile
        :param action_spec: action spec
        :param target_lane: target lane number
        :param lane_width: lane width
        :param T_d: the largest possible lateral time, may be bounded by safety
        :return: the action's cost and the cost components array (for debugging)
        """
        efficiency_cost = lon_comf_cost = lat_comf_cost = right_lane_cost = 0
        if action_spec.t > 0:
            efficiency_cost = BP_EfficiencyMetric.calc_cost(vel_profile)
            lon_comf_cost, lat_comf_cost = BP_ComfortMetric.calc_cost(ego_fstate, action_spec, T_d)
            right_lane_cost = BP_RightLaneMetric.calc_cost(action_spec.t, target_lane)

        # calculate maximal deviation from lane center
        signed_lat_dist = action_spec.d - ego_fstate[FS_DX]
        rel_lat = abs(signed_lat_dist)/lane_width
        rel_vel = ego_fstate[FS_DV]/lane_width
        if signed_lat_dist * rel_vel < 0:  # changes lateral direction
            rel_lat += rel_vel*rel_vel/(2*AGGRESSIVENESS_TO_LAT_ACC[0])  # predict maximal deviation
        max_lane_dev = min(2*rel_lat, 1)  # for half-lane deviation, max_lane_dev = 1
        lane_deviation_cost = BP_LaneDeviationMetric.calc_cost(max_lane_dev)
        return np.array([efficiency_cost, lon_comf_cost, lat_comf_cost, right_lane_cost, lane_deviation_cost])

    @staticmethod
    def _dist_to_target(state: BehavioralGridState, recipe: ActionRecipe):
        dist = np.inf
        if recipe.action_type != ActionType.FOLLOW_LANE:
            target_obj = state.road_occupancy_grid[(recipe.relative_lane, recipe.relative_lon)][0].dynamic_object
            # TODO: change it to the navigation frame state
            dist = target_obj.map_state.road_state[FS_SX] - state.ego_state.map_state.road_state[FS_SX]
        return dist

    @staticmethod
    def _get_lane_width(ego_fstate: np.array, ego_lane: int, center_lane_latitudes: np.array) -> float:
        if ego_fstate[FS_DX] <= center_lane_latitudes[ego_lane] and ego_lane > 0:
            return center_lane_latitudes[ego_lane] - center_lane_latitudes[ego_lane-1]
        elif ego_fstate[FS_DX] >= center_lane_latitudes[ego_lane] and ego_lane < len(center_lane_latitudes)-1:
            return center_lane_latitudes[ego_lane+1] - center_lane_latitudes[ego_lane]
        elif ego_lane > 0:  # ego_lat > center_lane_latitudes[-1]
            return center_lane_latitudes[ego_lane] - center_lane_latitudes[ego_lane - 1]
        elif ego_lane < len(center_lane_latitudes)-1:  # ego_lat < center_lane_latitudes[0]
            return center_lane_latitudes[ego_lane + 1] - center_lane_latitudes[ego_lane]
        else:  # single lane
            return 2*center_lane_latitudes[0]
