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
        ego = behavioral_state.ego_state
        ego_road = ego.road_localization
        ego_lane = ego_road.lane_num
        road_frenet = MapUtils.get_road_rhs_frenet(ego)
        ego_fstate = MapUtils.get_ego_road_localization(ego, road_frenet)
        lane_width = MapService.get_instance().get_road(ego_road.road_id).lane_width

        print('\ntime=%.1f ego_v=%.2f ego_lat=%.2f ego_dv=%.2f grid_size=%d' %
              (ego.timestamp_in_sec, ego.v_x, ego_road.intra_road_lat, ego_fstate[FS_DV], len(behavioral_state.road_occupancy_grid)))

        costs = np.full(len(action_recipes), np.inf)

        # loop over all specs / actions
        for i, spec in enumerate(action_specs):
            if not action_specs_mask[i]:
                continue

            recipe = action_recipes[i]
            target_lane = ego_lane + recipe.relative_lane.value
            lat_dist = (target_lane + 0.5) * lane_width - ego_road.intra_road_lat
            # calculate lateral time according to the CALM aggressiveness level
            T_d_calm = VelocityProfile.calc_lateral_time(ego_fstate[FS_DV], lat_dist, lane_width, AggressivenessLevel.CALM)

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

            # calculate actions costs
            sub_costs = HeuristicActionSpecEvaluator._calc_action_costs(ego_fstate, vel_profile, spec, lane_width, T_d_max)
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
        if recipe.action_type == ActionType.FOLLOW_VEHICLE or recipe.action_type == ActionType.OVERTAKE_VEHICLE:
            dist = spec.s - spec.t * spec.v - ego_fstate[FS_SX]
            return VelocityProfile.calc_profile_given_T(ego_fstate[FS_SV], spec.t, dist, spec.v)
        else:  # static action (FOLLOW_LANE)
            t2 = max(0., MINIMAL_STATIC_ACTION_TIME - spec.t)  # TODO: remove it after implementation of value function
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
        ego_road = behavioral_state.ego_state.road_localization
        ego_lon = ego_road.road_lon
        cur_time = behavioral_state.ego_state.timestamp_in_sec

        forward_cell = (action_lat_cell, RelativeLongitudinalPosition.FRONT)
        front_cell = (RelativeLane.SAME_LANE, RelativeLongitudinalPosition.FRONT)
        side_rear_cell = (action_lat_cell, RelativeLongitudinalPosition.REAR)
        rear_cell = (RelativeLane.SAME_LANE, RelativeLongitudinalPosition.REAR)

        lat_dist_to_target = abs(action_lat_cell.value - (ego_road.intra_lane_lat / lane_width - 0.5))  # in [0, 1.5]
        # increase time delay if ego does not move laterally according to the current action
        is_moving_laterally_to_target = (action_lat_cell != RelativeLane.SAME_LANE and
                                         ego_road.intra_road_yaw * action_lat_cell.value <= 0)

        # check safety w.r.t. the followed object on the target lane (if exists)
        if forward_cell in behavioral_state.road_occupancy_grid:
            followed_obj = behavioral_state.road_occupancy_grid[forward_cell][0].dynamic_object

            td = AV_TIME_DELAY
            if action.action_type == ActionType.FOLLOW_LANE:
                td = SAFE_DIST_TIME_DELAY  # TODO: remove it after value function implementation

            # calculate last safe time w.r.t. the followed object
            forward_safe_time = vel_profile.calc_last_safe_time(ego_lon, ego_length,
                    followed_obj.road_localization.road_lon, followed_obj.v_x, followed_obj.size.length, np.inf, td, td)
            if forward_safe_time < vel_profile.total_time():
                obj_lon = followed_obj.road_localization.road_lon
                act_time = vel_profile.total_time()
                print('forward unsafe: %d(%d %d) rel_lat=%d dist=%.2f t=%.2f final_dist=%.2f v_obj=%.2f '
                      'prof=(t=[%.2f %.2f %.2f] v=[%.2f %.2f %.2f]) safe_time=%.2f td=%.2f' %
                      (i, action.action_type.value, action.aggressiveness.value, action_lat_cell.value, obj_lon - ego_lon,
                       act_time, obj_lon + act_time * followed_obj.v_x - (ego_lon + vel_profile.total_dist()),
                       followed_obj.v_x, vel_profile.t1, vel_profile.t2, vel_profile.t3, vel_profile.v_init,
                       vel_profile.v_mid, vel_profile.v_tar, forward_safe_time, td))
                return -1

        safe_time = np.inf
        if action_lat_cell != RelativeLane.SAME_LANE:  # lane change action
            # TODO: move it to a filter
            # check whether there is a car in the neighbor cell (same longitude)
            if (action_lat_cell, RelativeLongitudinalPosition.PARALLEL) in behavioral_state.road_occupancy_grid:
                print('side unsafe rel_lat=%d: action %d' % (action_lat_cell.value, i))
                return -1

            # check safety w.r.t. the front object F on the original lane (if exists)
            if front_cell in behavioral_state.road_occupancy_grid:
                front_obj = behavioral_state.road_occupancy_grid[front_cell][0].dynamic_object
                # time delay decreases as function of lateral distance to the target: td_0 = td_T + 1
                # td_0 > td_T, since as latitude advances ego can escape laterally easier
                td_T = 0.  # dist to F after completing lane change. TODO: increase it when the planning will be deep
                td_0 = AV_TIME_DELAY * lat_dist_to_target
                # calculate last safe time w.r.t. F
                front_safe_time = vel_profile.calc_last_safe_time(ego_lon, ego_length,
                    front_obj.road_localization.road_lon, front_obj.v_x, front_obj.size.length, 0.75 * T_d, td_0, td_T)
                if front_safe_time < np.inf:
                    print('front_safe_time=%.2f action %d(%d %d): front_dist=%.2f front_vel=%.2f lat_d=%.2f td_0=%.2f td_T=%.2f' %
                          (front_safe_time, i, action.action_type.value, action.aggressiveness.value,
                           front_obj.road_localization.road_lon - ego_lon, front_obj.v_x, lat_dist_to_target, td_0, td_T))
                if front_safe_time <= 0:
                    return -1
                safe_time = min(safe_time, front_safe_time)

            # check safety w.r.t. the back object on the original lane (if exists)
            if side_rear_cell in behavioral_state.road_occupancy_grid:
                back_obj = behavioral_state.road_occupancy_grid[side_rear_cell][0].dynamic_object
                td = SAFE_DIST_TIME_DELAY
                # calculate last safe time w.r.t. LB or RB
                back_safe_time = vel_profile.calc_last_safe_time(ego_lon, ego_length,
                        back_obj.road_localization.road_lon, back_obj.v_x, back_obj.size.length, T_d, td, td)
                if back_safe_time < np.inf:
                    print('back_safe_time=%.2f action %d(%d %d): back_dist=%.2f back_vel=%.2f rel_lat=%.2f td=%.2f' %
                          (back_safe_time, i, action.action_type.value, action.aggressiveness.value,
                           ego_lon - back_obj.road_localization.road_lon, back_obj.v_x, action_lat_cell.value, td))
                # if ego is unsafe w.r.t. back_obj, then save a flag for the case ego will enter to its lane,
                # such that ego will check safety w.r.t to the rear object
                if back_safe_time <= 0 and is_moving_laterally_to_target:
                    self.back_danger_lane = ego_road.lane_num + action_lat_cell.value
                    self.back_danger_side = action_lat_cell.value  # -1 or 1
                    self.back_danger_time = cur_time
                if back_safe_time <= 0:
                    return -1
                safe_time = min(safe_time, back_safe_time)

        # check safety w.r.t. the rear object R for the case we are after back danger and arrived to the dangerous lane
        if self.back_danger_lane is not None:
            if cur_time - self.back_danger_time < 4:  # the danger is still relevant
                lat_dist = self.back_danger_lane - (ego_road.intra_road_lat/lane_width - 0.5)
                if 0.25 < abs(lat_dist) < 0.5 and lat_dist * self.back_danger_side > 0 and \
                                     action_lat_cell.value * self.back_danger_side >= 0:
                    if rear_cell in behavioral_state.road_occupancy_grid:
                        td = SAFE_DIST_TIME_DELAY
                        rear_obj = behavioral_state.road_occupancy_grid[rear_cell][0].dynamic_object
                        # calculate last safe time w.r.t. R
                        back_safe_time = vel_profile.calc_last_safe_time(ego_lon, ego_length,
                                rear_obj.road_localization.road_lon, rear_obj.v_x, rear_obj.size.length, T_d, td, td)
                        if back_safe_time <= 0:
                            return -1
                        safe_time = min(safe_time, back_safe_time)
            else:  # after timeout, delete the danger flag
                self.back_danger_lane = None

        # print('front_time=%f back_time=%f forward_time=%f safe_time=%f' % \
        # (front_safe_time, back_safe_time, forward_safe_time, safe_time))
        return safe_time

    @staticmethod
    def _calc_action_costs(ego_fstate: np.array, vel_profile: VelocityProfile, action_spec: ActionSpec,
                           lane_width: float, T_d: float) -> [float, np.array]:
        """
        Calculate the cost of the action
        :param vel_profile: longitudinal velocity profile
        :param action_spec: action spec
        :param lane_width: lane width
        :param T_d_max: the largest possible lateral time, may be bounded by safety
        :return: the action's cost and the cost components array (for debugging)
        """
        # calculate efficiency, comfort and non-right lane costs
        efficiency_cost = lon_comf_cost = lat_comf_cost = right_lane_cost = 0
        target_lane = int(action_spec.d / lane_width)
        efficiency_cost = BP_EfficiencyMetric.calc_cost(vel_profile)
        lon_comf_cost, lat_comf_cost = BP_ComfortMetric.calc_cost(ego_fstate, action_spec, T_d)
        right_lane_cost = BP_RightLaneMetric.calc_cost(action_spec.t, target_lane)

        # calculate maximal deviation from lane center for lane deviation cost
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
        """
        given action recipe, calculate longitudinal distance from the target object, and inf for static action
        :param state: behavioral state
        :param recipe: action recipe
        :return: distance from the target
        """
        dist = np.inf
        if recipe.action_type != ActionType.FOLLOW_LANE:
            target_obj = state.road_occupancy_grid[(recipe.relative_lane, recipe.relative_lon)][0].dynamic_object
            dist = target_obj.road_localization.road_lon - state.ego_state.road_localization.road_lon
        return dist
