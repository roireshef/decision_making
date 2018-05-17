from logging import Logger
from typing import List, Optional

import numpy as np
import sys

from decision_making.src.global_constants import BP_METRICS_TIME_HORIZON, \
    BP_EFFICIENCY_COST_WEIGHT, WERLING_TIME_RESOLUTION, BP_RIGHT_LANE_COST_WEIGHT, \
    BP_METRICS_LANE_DEVIATION_COST_WEIGHT, BP_MISSING_GOAL_COST, BP_MAX_VELOCITY_TOLERANCE, SAFE_DIST_TIME_DELAY, \
    BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED, AGGRESSIVENESS_TO_LON_ACC, AV_TIME_DELAY
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState, \
    RelativeLongitudinalPosition
from decision_making.src.planning.behavioral.data_objects import ActionRecipe, ActionType, \
    AggressivenessLevel, RelativeLane, NavigationGoal, ActionSpec
from decision_making.src.planning.behavioral.evaluators.action_evaluator import ActionSpecEvaluator
from decision_making.src.planning.behavioral.evaluators.cost_functions import \
    BP_EfficiencyMetric, \
    BP_ComfortMetric, BP_RightLaneMetric, VelocityProfile, BP_LaneDeviationMetric
from decision_making.src.planning.types import FP_SX
from decision_making.src.prediction.predictor import Predictor
from decision_making.src.state.state import EgoState, State
from mapping.src.model.localization import RoadLocalization
from mapping.src.service.map_service import MapService


class HeuristicActionSpecEvaluator(ActionSpecEvaluator):
    def __init__(self, logger: Logger):
        super().__init__(logger)
        self.last_action_lane = None
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
        :param goal: navigation goal
        :return: array of costs (one cost per action)
        """
        ego = behavioral_state.ego_state
        ego_lane = ego.road_localization.lane_num
        if self.last_action_lane is None:
            self.last_action_lane = ego_lane
        ego_lat_vel = ego.v_x * np.sin(ego.road_localization.intra_road_yaw)
        lane_width = MapService.get_instance().get_road(ego.road_localization.road_id).lane_width

        # calculate full lane change time (calm)
        T_d_full, _, _ = VelocityProfile.calc_lateral_time(0, lane_width, lane_width, AggressivenessLevel.CALM)
        calm_lat_comfort_cost = BP_ComfortMetric.calc_cost(
            VelocityProfile(0, 0, 0, 0, 0, 0), T_d_full, T_d_full, AggressivenessLevel.CALM, 0)

        # get front dynamic objects' velocities from the occupancy grid
        front_objects = self._get_front_objects(behavioral_state)

        print('\ntime=%.1f ego_v=%.2f ego_lat=%.2f ego_dv=%.2f grid_size=%d' % (ego.timestamp_in_sec, ego.v_x,
                ego.road_localization.intra_road_lat, ego_lat_vel, len(behavioral_state.road_occupancy_grid)))

        action_costs = np.full(len(action_recipes), np.inf)

        for i, action_spec in enumerate(action_specs):
            if not action_specs_mask[i]:
                continue

            action_recipe = action_recipes[i]
            target_lane = ego_lane + action_recipe.relative_lane.value
            lat_dist = (target_lane + 0.5) * lane_width - ego.road_localization.intra_road_lat
            # calculate lateral time according to the given aggressiveness level
            T_d, lat_dev, lat_vel_to_tar = VelocityProfile.calc_lateral_time(ego_lat_vel, lat_dist, lane_width,
                                                                             action_recipe.aggressiveness)

            # create velocity profile, whose length is at least as the lateral time
            vel_profile = self._calc_velocity_profile(behavioral_state, action_recipe, action_spec, front_objects)
            #print('version is: %s' % sys.version)
            if vel_profile is None:  # infeasible action
                continue

            # calculate the latest safe time
            T_d_max = self._calc_largest_safe_time(behavioral_state, action_recipe, i, vel_profile, ego.size.length/2,
                                                   T_d, lane_width)
            if T_d_max <= 0:
                continue

            action_costs[i], sub_costs = self._calc_action_cost(
                ego, vel_profile, target_lane, T_d, T_d_max, T_d_full, lat_dev, lat_vel_to_tar, lane_width,
                action_recipe.aggressiveness, calm_lat_comfort_cost)

            # FOR DEBUGGING
            dist = np.inf
            if action_recipe.action_type != ActionType.FOLLOW_LANE:
                target_obj = behavioral_state.road_occupancy_grid[(action_recipe.relative_lane, action_recipe.relative_lon)][0].dynamic_object
                dist = target_obj.road_localization.road_lon - ego.road_localization.road_lon
            print(
                'action %d(%d) lane %d: dist=%.1f tar_vel=%.2f [eff %.3f comf %.3f right %.2f dev %.2f value %.2f] [T_d=%.2f Tdmax=%.2f prof_time=%.2f]: tot %.2f' %
                (i, action_recipe.action_type.value, target_lane, dist, vel_profile.v_tar,
                 sub_costs[0], sub_costs[1], sub_costs[2], sub_costs[3], sub_costs[4], T_d, T_d_max,
                 vel_profile.total_time(), action_costs[i]))
            # print('vel_prof {t1=%.2f t2=%.2f t3=%.2f v_init=%.2f v_mid=%.2f v_tar=%.2f}' % (vel_profile.t1, vel_profile.t2, vel_profile.t3, vel_profile.v_init, vel_profile.v_mid, vel_profile.v_tar))

        best_action = int(np.argmin(action_costs))
        self.last_action_lane = ego_lane + action_recipes[best_action].relative_lane.value

        if np.isinf(np.min(action_costs)):
            print('************************************************************')
            print('********************  NO SAFE ACTION  **********************')
            print('************************************************************')
        else:
            print('Best action %d; lane %d\n' % (best_action, ego_lane + action_recipes[best_action].relative_lane.value))

        return action_costs

    # TODO: remove this function (used by hack) after implementation of smarter value function
    def _get_front_objects(self, behavioral_state: BehavioralGridState):
        """
        given occupancy grid, retrieve at most 3 front objects (from 3 relevant lanes)
        :param behavioral_state: behavioral state containing the occupancy grid
        :return: array of the front objects. None for missing object
        """
        # get front dynamic objects from the occupancy grid
        front_objects = {}
        for lat in [RelativeLane.RIGHT_LANE, RelativeLane.SAME_LANE, RelativeLane.LEFT_LANE]:
            if (lat, RelativeLongitudinalPosition.FRONT) in behavioral_state.road_occupancy_grid:
                front_objects[lat] = behavioral_state.road_occupancy_grid[(lat, RelativeLongitudinalPosition.FRONT)][0].dynamic_object
        return front_objects

    def _calc_velocity_profile(self, behavioral_state: BehavioralGridState, action_recipe: ActionRecipe,
                               action_spec: ActionSpec, front_objects: np.array) -> Optional[VelocityProfile]:
        """
        Given action recipe and behavioral state, calculate the longitudinal velocity profile for that action.
        :param behavioral_state:
        :param action_recipe: the input action
        :param T_d: the action's lateral time (as a lower bound for the vel_profile total time)
        :param front_objects: array of the front objects from the occupancy grid
        :return: longitudinal velocity profile or None if the action is infeasible by given aggressiveness level
        """
        ego = behavioral_state.ego_state
        ego_fpoint = np.array([ego.road_localization.road_lon, ego.road_localization.intra_road_lat])
        target_vel = action_spec.v

        if action_recipe.action_type == ActionType.FOLLOW_VEHICLE or action_recipe.action_type == ActionType.OVER_TAKE_VEHICLE:
            target_obj = behavioral_state.road_occupancy_grid[(action_recipe.relative_lane, action_recipe.relative_lon)][0].dynamic_object
            obj_lon = target_obj.road_localization.road_lon
            cars_size_margin = 0.5 * (ego.size.length + target_obj.size.length)

            safe_dist = SAFE_DIST_TIME_DELAY * target_vel + cars_size_margin
            lon_diff = obj_lon - ego_fpoint[FP_SX]
            if action_recipe.action_type == ActionType.FOLLOW_VEHICLE:
                return VelocityProfile.calc_profile_given_T(ego.v_x, action_spec.t, lon_diff - safe_dist, target_vel)
            else:  # action_recipe.action_type == ActionType.OVER_TAKE_VEHICLE:
                return VelocityProfile.calc_profile_given_T(ego.v_x, action_spec.t, lon_diff + safe_dist, target_vel)

        else:  # static action (FOLLOW_LANE)

            # TODO: remove this hack after implementation of smarter value function
            # skip static action if its velocity is greater than velocity of the front object on the same lane,
            # since it's cost always will be better than goto_left_lane, regardless F_vel.
            # This condition will be removed when a real value function will be used.
            if action_recipe.relative_lane in front_objects:
                front_obj = front_objects[action_recipe.relative_lane]
                acc = AGGRESSIVENESS_TO_LON_ACC[action_recipe.aggressiveness.value]
                dist = front_obj.road_localization.road_lon - ego.road_localization.road_lon
                if VelocityProfile.calc_collision_time(ego.v_x, target_vel, acc, front_obj.v_x, dist) < 20:
                # if target_vel > front_obj.v_x + BP_MAX_VELOCITY_TOLERANCE:
                    if target_vel <= BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED:
                        print('filter fast stat_act: rel_lat=%d, v_tar=%.2f, front.v_x=%.2f' %
                              (action_recipe.relative_lane.value, target_vel, front_obj.v_x))
                    return None

            return VelocityProfile(v_init=ego.v_x, t1=action_spec.t, v_mid=target_vel, t2=0, t3=0, v_tar=target_vel)

    def _calc_largest_safe_time(self, behavioral_state: BehavioralGridState, action: ActionRecipe, i: int,
                                vel_profile: VelocityProfile, ego_half_size: float, T_d: float, lane_width: float) -> float:
        """
        For a lane change action, given ego velocity profile and behavioral_state, get two cars that may
        require faster lateral movement (the front overtaken car and the back interfered car) and calculate the last
        time, for which the safety holds w.r.t. these two cars.
        :param behavioral_state: semantic actions grid behavioral state
        :param vel_profile: the velocity profile of ego
        :param ego_half_size: half ego length
        :param T_d: time for comfortable lane change
        :param lane_width: lane width of the road
        :return: the latest time, when ego is still safe
        """
        action_lat_cell = action.relative_lane
        ego_road = behavioral_state.ego_state.road_localization
        ego_lon = ego_road.road_lon
        action_lane = ego_road.lane_num + action_lat_cell.value
        cur_time = behavioral_state.ego_state.timestamp_in_sec
        lat_dist_to_target = abs(action_lat_cell.value - (ego_road.intra_lane_lat / lane_width - 0.5))  # in [0, 1.5]

        change_tar_lane_td = 0.2 * abs(action_lane - self.last_action_lane)  # addition to time delay
        big_follow_td = AV_TIME_DELAY + 0.8  # follow lane TODO: remove it after value function implementation
        min_front_td = 0.0  # dist to F after completing lane change. TODO: increase it when the planning will be deep

        # check safety w.r.t. the front object on the target lane (if exists)
        if (action_lat_cell, RelativeLongitudinalPosition.FRONT) in behavioral_state.road_occupancy_grid:
            followed_obj = behavioral_state.road_occupancy_grid[(action_lat_cell, RelativeLongitudinalPosition.FRONT)][0].dynamic_object

            # TODO: temp hack: smaller delay in following front car, to enable calm distance increasing if dist < 2 sec
            # TODO: the hack should be removed after value function implementation. Then td = AV_TIME_DELAY
            # without the hack this action is unsafe and filtered, and ego must move to another lane
            if action.action_type == ActionType.FOLLOW_VEHICLE:
                td = AV_TIME_DELAY
            else:
                td = big_follow_td
            td += 2 * change_tar_lane_td

            forward_safe_time = vel_profile.calc_last_safe_time(ego_lon, ego_half_size, followed_obj, np.inf, td, td)
            if forward_safe_time < vel_profile.total_time():
                obj_lon = followed_obj.road_localization.road_lon
                act_time = vel_profile.total_time()
                print('forward unsafe: act_type %d, rel_lat %d, dist=%.2f, act_time=%.2f final_dist=%.2f v_obj=%.2f '
                      'prof=(t=[%.2f %.2f %.2f] v=[%.2f %.2f %.2f], safe_time=%.2f td=%.2f: action %d)' %
                      (action.action_type.value, action_lat_cell.value, obj_lon - ego_lon,
                       act_time, obj_lon + act_time * followed_obj.v_x - (ego_lon + vel_profile.total_dist()),
                       followed_obj.v_x, vel_profile.t1, vel_profile.t2, vel_profile.t3, vel_profile.v_init,
                       vel_profile.v_mid, vel_profile.v_tar, forward_safe_time, td, i))
                return 0

        safe_time = np.inf
        if action_lat_cell != RelativeLane.SAME_LANE:
            # TODO: move it to a filter
            # check whether there is a car in the neighbor cell (same longitude)
            if (action_lat_cell, RelativeLongitudinalPosition.PARALLEL) in behavioral_state.road_occupancy_grid:
                print('side unsafe rel_lat=%d: action %d' % (action_lat_cell.value, i))
                return 0

            # check safety w.r.t. the front object on the original lane (if exists)
            if (RelativeLane.SAME_LANE, RelativeLongitudinalPosition.FRONT) in behavioral_state.road_occupancy_grid:
                front_obj = behavioral_state.road_occupancy_grid[
                    (RelativeLane.SAME_LANE, RelativeLongitudinalPosition.FRONT)][0].dynamic_object
                # time delay decreases as function of lateral distance to the target: td_0 = td_T + 1
                # td_0 > td_T, since as latitude advances ego can escape laterally easier
                td_T = min_front_td + change_tar_lane_td
                td_0 = AV_TIME_DELAY * lat_dist_to_target + change_tar_lane_td
                front_safe_time = vel_profile.calc_last_safe_time(ego_lon, ego_half_size, front_obj, T_d, td_0, td_T)
                if front_safe_time < np.inf:
                    print('front_safe_time=%.2f front_dist=%.2f front_vel=%.2f lat_d=%.2f td_0=%.2f td_T=%.2f: action %d' %
                          (front_safe_time, front_obj.road_localization.road_lon - ego_lon, front_obj.v_x,
                           lat_dist_to_target, td_0, td_T, i))
                if front_safe_time <= 0:
                    return front_safe_time
                safe_time = min(safe_time, front_safe_time)

            # check safety w.r.t. the back object on the original lane (if exists)
            if (action_lat_cell, RelativeLongitudinalPosition.REAR) in behavioral_state.road_occupancy_grid:
                back_obj = behavioral_state.road_occupancy_grid[(action_lat_cell, RelativeLongitudinalPosition.REAR)][0].dynamic_object
                td = SAFE_DIST_TIME_DELAY + 3*change_tar_lane_td
                back_safe_time = vel_profile.calc_last_safe_time(ego_lon, ego_half_size, back_obj, T_d, td, td)
                if back_safe_time < np.inf:
                    print('back_safe_time=%.2f back_dist=%.2f back_vel=%.2f rel_lat=%.2f td=%.2f: action %d' %
                          (back_safe_time, ego_lon - back_obj.road_localization.road_lon, back_obj.v_x,
                           action_lat_cell.value, td, i))
                safe_time = min(safe_time, back_safe_time)
                # if ego is unsafe w.r.t. back_obj, then save a flag for the case ego will enter to its lane,
                # such that ego will check safety w.r.t to the rear object
                if back_safe_time <= 0 and action_lane == self.last_action_lane:
                    self.back_danger_lane = ego_road.lane_num + action_lat_cell.value
                    self.back_danger_side = action_lat_cell.value  # -1 or 1
                    self.back_danger_time = cur_time

        # check safety w.r.t. the rear object R for the case we are after back danger and arrived to the dangerous lane
        if self.back_danger_lane is not None:
            if cur_time - self.back_danger_time < 4:  # the danger is still relevant
                lat_dist = self.back_danger_lane - (ego_road.intra_road_lat/lane_width - 0.5)
                if 0.25 < abs(lat_dist) < 0.5 and lat_dist * self.back_danger_side > 0 and \
                                     action_lat_cell.value * self.back_danger_side >= 0:
                    rear_cell = (RelativeLane.SAME_LANE, RelativeLongitudinalPosition.REAR)
                    if rear_cell in behavioral_state.road_occupancy_grid:
                        td = SAFE_DIST_TIME_DELAY
                        rear_obj = behavioral_state.road_occupancy_grid[rear_cell][0].dynamic_object
                        back_safe_time = vel_profile.calc_last_safe_time(ego_lon, ego_half_size, rear_obj, T_d, td, td)
                        safe_time = min(safe_time, back_safe_time)
            else:  # after timeout, delete the danger flag
                self.back_danger_lane = None

        # print('front_time=%f back_time=%f forward_time=%f safe_time=%f' % \
        # (front_safe_time, back_safe_time, forward_safe_time, safe_time))
        return safe_time

    def _calc_action_cost(self, ego: EgoState, vel_profile: VelocityProfile, target_lane: int,
                          T_d: float, T_d_max: float, T_d_full: float, lat_dev: float, lat_vel_to_tar: float,
                          lane_width: float, aggressiveness: AggressivenessLevel, calm_lat_comfort_cost: float) -> \
            [float, np.array]:
        """
        Calculate the cost of the action
        :param vel_profile: longitudinal velocity profile
        :param target_lane: target lane index
        :param T_d: lateral time according to the given aggressiveness level
        :param T_d_max: largest lateral time limited by safety
        :param T_d_full: lateral time of full lane change by calm action
        :param lat_dev: maximal lateral deviation from lane center (for half lane deviation, lat_dev = 1)
        :param lat_vel_to_tar: initial lateral velocity to the target direction
        :param aggressiveness: aggressiveness level of the action
        :param goal: navigation goal
        :return: the action's cost and the cost components array (for debugging)
        """
        vel_profile_time = vel_profile.total_time()
        efficiency_cost = comfort_cost = right_lane_cost = 0
        if vel_profile_time > 0:
            efficiency_cost = BP_EfficiencyMetric.calc_cost(vel_profile)
            comfort_cost = BP_ComfortMetric.calc_cost(vel_profile, T_d, T_d_max, aggressiveness, lat_vel_to_tar)
            right_lane_cost = BP_RightLaneMetric.calc_cost(vel_profile_time, target_lane)
        lane_deviation_cost = BP_LaneDeviationMetric.calc_cost(lat_dev)

        new_ego_localization = RoadLocalization(ego.road_localization.road_id, target_lane, target_lane * lane_width, 0,
                                                ego.road_localization.road_lon + vel_profile.total_dist(), 0)

        value_function = HeuristicActionSpecEvaluator.value_function_approximation(
            new_ego_localization, vel_profile.v_tar, BP_METRICS_TIME_HORIZON - vel_profile_time, T_d_full,
            calm_lat_comfort_cost, None)

        return efficiency_cost + comfort_cost + right_lane_cost + lane_deviation_cost + value_function, \
               np.array([efficiency_cost, comfort_cost, right_lane_cost, lane_deviation_cost, value_function])

    @staticmethod
    def value_function_approximation(road_localization: RoadLocalization, v_tar: float, time_period: float,
                                     T_d_full: float, calm_lat_comfort_cost: float, goal: Optional[NavigationGoal]) -> float:
        # TODO: temporary function until value function is not implemented
        ego_road_id = road_localization.road_id
        ego_lane = road_localization.lane_num
        ego_lon = road_localization.road_lon

        time_period = max(0., time_period)
        efficiency_cost = right_lane_cost = future_lane_deviation_cost = future_comfort_cost = goal_cost = 0
        if time_period > 0:
            efficiency_cost = BP_EfficiencyMetric.calc_pointwise_cost_for_velocities(np.array([v_tar]))[0] * \
                              BP_EFFICIENCY_COST_WEIGHT * time_period
            right_lane_cost = ego_lane * time_period * BP_RIGHT_LANE_COST_WEIGHT

        if goal is None or ego_road_id != goal.road_id:  # no relevant goal
            if ego_lane > 0:  # distance in lanes from the rightest lane
                goal_cost = ego_lane * (BP_METRICS_LANE_DEVIATION_COST_WEIGHT + calm_lat_comfort_cost)
                future_lane_deviation_cost = future_comfort_cost = 0
        elif len(goal.lanes_list) > 0 and ego_lane not in goal.lanes_list:  # outside of the lanes range of the goal
            if ego_lon >= goal.lon:  # we missed the goal
                goal_cost = BP_MISSING_GOAL_COST
            else:  # still did not arrive to the goal
                lanes_from_goal = np.min(abs(np.array(goal.lanes_list) - ego_lane))
                T_d_max_per_lane = np.inf
                if v_tar * lanes_from_goal > 0:
                    T_d_max_per_lane = (goal.lon - ego_lon) / (
                    v_tar * lanes_from_goal)  # required time for one lane change
                empty_vel_profile = VelocityProfile(0, 0, 0, 0, 0, 0)
                goal_comfort_cost = lanes_from_goal * BP_ComfortMetric.calc_cost(
                    empty_vel_profile, T_d_full, T_d_max_per_lane, AggressivenessLevel.CALM, 0.)
                goal_cost = min(BP_MISSING_GOAL_COST, goal_comfort_cost)

        cost = efficiency_cost + future_comfort_cost + right_lane_cost + future_lane_deviation_cost + goal_cost
        return cost
