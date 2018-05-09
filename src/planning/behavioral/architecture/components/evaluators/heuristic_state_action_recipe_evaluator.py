from logging import Logger
from typing import List, Optional

import numpy as np
import sys

from decision_making.src.global_constants import SEMANTIC_CELL_LON_FRONT, SEMANTIC_CELL_LON_SAME, \
    SEMANTIC_CELL_LAT_SAME, BP_METRICS_TIME_HORIZON, SEMANTIC_CELL_LAT_RIGHT, SEMANTIC_CELL_LAT_LEFT, \
    EFFICIENCY_COST_WEIGHT, WERLING_TIME_RESOLUTION, RIGHT_LANE_COST_WEIGHT, \
    BP_METRICS_LANE_DEVIATION_COST_WEIGHT, BP_MISSING_GOAL_COST, BP_MAX_VELOCITY_TOLERANCE, SAFE_DIST_TIME_DELAY, \
    BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED
from decision_making.src.global_constants import SEMANTIC_CELL_LON_REAR
from decision_making.src.planning.behavioral.architecture.components.evaluators.cost_functions import \
    PlanEfficiencyMetric, \
    PlanComfortMetric, PlanRightLaneMetric, VelocityProfile, PlanLaneDeviationMetric
from decision_making.src.planning.behavioral.architecture.components.evaluators.state_action_evaluator import \
    StateActionRecipeEvaluator
from decision_making.src.planning.behavioral.architecture.components.evaluators.value_approximator import \
    ValueApproximator
from decision_making.src.planning.behavioral.architecture.components.evaluators.velocity_profile import ProfileSafety
from decision_making.src.planning.behavioral.architecture.data_objects import ActionRecipe, ActionType, \
    AggressivenessLevel, RelativeLane, NavigationGoal
from decision_making.src.planning.behavioral.architecture.semantic_behavioral_grid_state import \
    SemanticBehavioralGridState
from decision_making.src.planning.types import FP_SX, FP_DX, FS_SV, FS_SA
from decision_making.src.state.state import EgoState
from mapping.src.model.localization import RoadLocalization
from mapping.src.service.map_service import MapService


class HeuristicStateActionRecipeEvaluator(StateActionRecipeEvaluator):
    def __init__(self, logger: Logger):
        super().__init__(logger)

    def evaluate_recipes(self, behavioral_state: SemanticBehavioralGridState, action_recipes: List[ActionRecipe],
                         action_recipes_mask: List[bool], goal: Optional[NavigationGoal]) -> np.ndarray:
        """
        Gets a list of actions to evaluate and returns a vector representing their costs.
        A set of actions is provided, enabling us to assess them independently.
        Note: the semantic actions were generated using the behavioral state and don't necessarily capture
        all relevant details in the scene. Therefore the evaluation is done using the behavioral state.
        :param behavioral_state: semantic actions grid behavioral state
        :param action_recipes: array of semantic actions
        :param action_recipes_mask: array of boolean values indicating whether the recipe was filtered or not
        :param goal: navigation goal
        :return: array of costs (one cost per action)
        """
        ego = behavioral_state.ego_state
        ego_lane = ego.road_localization.lane_num
        ego_lat_vel = ego.v_x * np.sin(ego.road_localization.intra_road_yaw)
        lane_width = MapService.get_instance().get_road(ego.road_localization.road_id).lane_width

        # calculate full lane change time (calm)
        T_d_full, _ = VelocityProfile.calc_lateral_time(0, lane_width, lane_width, AggressivenessLevel.CALM)

        # get front dynamic objects' velocities from the occupancy grid
        front_objects = self._get_front_objects(behavioral_state)

        print('time=%.1f ego_v=%.2f ego_lat=%.2f ego_dv=%.2f' % (ego.timestamp_in_sec, ego.v_x,
                                                                 ego.road_localization.intra_road_lat, ego_lat_vel))

        action_costs = np.full(len(action_recipes), np.inf)

        for i, action_recipe in enumerate(action_recipes):
            if not action_recipes_mask[i]:
                continue

            target_lane = ego_lane + action_recipe.relative_lane.value
            lat_dist = (target_lane + 0.5) * lane_width - ego.road_localization.intra_road_lat
            # calculate lateral time according to the given aggressiveness level
            T_d, lat_dev = VelocityProfile.calc_lateral_time(ego_lat_vel, lat_dist, lane_width, action_recipe.aggressiveness)

            # create velocity profile, whose length is at least as the lateral time
            vel_profile = self._calc_velocity_profile(behavioral_state, action_recipe, T_d, front_objects)
            #print('version is: %s' % sys.version)
            if vel_profile is None:  # infeasible action
                continue

            # calculate the latest safe time
            T_d_max = self._calc_largest_safe_time(behavioral_state, action_recipe, vel_profile, ego.size.length / 2, T_d)
            if T_d_max <= 0:
                continue

            action_costs[i], sub_costs = self._calc_action_cost(ego, vel_profile, target_lane, T_d, T_d_max, T_d_full,
                                                                lat_dev, lane_width, action_recipe.aggressiveness, goal)

            # FOR DEBUGGING
            dist = np.inf
            if action_recipe.action_type != ActionType.FOLLOW_LANE:
                target_obj = behavioral_state.road_occupancy_grid[(action_recipe.relative_lane.value, action_recipe.relative_lon.value)][0]
                dist = target_obj.road_localization.road_lon - ego.road_localization.road_lon
            print('action %d(%d) lane %d: dist=%.1f tar_vel=%.2f [eff %.3f comf %.3f right %.2f dev %.2f value %.2f] [T_d=%.2f Tdmax=%.2f prof_time=%.2f]: tot %.2f' %
                  (i, action_recipe.action_type.value, target_lane, dist, vel_profile.v_tar,
                   sub_costs[0], sub_costs[1], sub_costs[2], sub_costs[3], sub_costs[4], T_d, T_d_max,
                   vel_profile.total_time(), action_costs[i]))
            #print('vel_prof {t1=%.2f t2=%.2f t3=%.2f v_init=%.2f v_mid=%.2f v_tar=%.2f}' % (vel_profile.t1, vel_profile.t2, vel_profile.t3, vel_profile.v_init, vel_profile.v_mid, vel_profile.v_tar))

        # FOR DEBUGGING
        best_action = int(np.argmin(action_costs))
        print('Best action %d; lane %d\n' % (best_action, ego_lane + action_recipes[best_action].relative_lane.value))

        return action_costs

    def _get_front_objects(self, behavioral_state: SemanticBehavioralGridState) -> np.array:
        """
        given occupancy grid, retrieve at most 3 front objects (from 3 relevant lanes)
        :param behavioral_state: behavioral state containing the occupancy grid
        :return: array of the front objects. None for missing object
        """
        # get front dynamic objects from the occupancy grid
        front_objects = np.array([None, None, None])
        for lat in [SEMANTIC_CELL_LAT_RIGHT, SEMANTIC_CELL_LAT_SAME, SEMANTIC_CELL_LAT_LEFT]:
            if (lat, SEMANTIC_CELL_LON_FRONT) in behavioral_state.road_occupancy_grid:
                front_objects[lat - SEMANTIC_CELL_LAT_RIGHT] = \
                    behavioral_state.road_occupancy_grid[(lat, SEMANTIC_CELL_LON_FRONT)][0]
        return front_objects

    def _calc_velocity_profile(self, behavioral_state: SemanticBehavioralGridState, action_recipe: ActionRecipe,
                               T_d: float, front_objects: np.array) -> Optional[VelocityProfile]:
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
        target_acc = cars_size_margin = obj_lon = None

        if action_recipe.action_type == ActionType.FOLLOW_VEHICLE or action_recipe.action_type == ActionType.TAKE_OVER_VEHICLE:
            target_obj = behavioral_state.road_occupancy_grid[
                (action_recipe.relative_lane.value, action_recipe.relative_lon.value)][0]
            target_vel = target_obj.v_x
            target_acc = target_obj.acceleration_lon
            obj_lon = target_obj.road_localization.road_lon
            cars_size_margin = 0.5 * (ego.size.length + target_obj.size.length)
        else:  # static action (FOLLOW_LANE)
            target_vel = action_recipe.velocity

            # TODO: remove this hack after implementation of smarter value function
            # skip static action if its velocity is greater than velocity of the front object on the same lane,
            # since it's cost always will be better than goto_left_lane, regardless F_vel.
            # This condition will be removed when a real value function will be used.
            front_obj = front_objects[action_recipe.relative_lane.value - SEMANTIC_CELL_LAT_RIGHT]
            if front_obj is not None:
                if target_vel > front_obj.v_x + BP_MAX_VELOCITY_TOLERANCE:
                    if target_vel <= BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED:
                        print('filter fast stat_act: rel_lat=%d, v_tar=%.2f, front.v_x=%.2f' %
                              (action_recipe.relative_lane.value, target_vel, front_obj.v_x))
                    return None

        vel_profile = VelocityProfile.calc_velocity_profile_given_acc(
            action_recipe.action_type, ego_fpoint[FP_SX], ego.v_x, obj_lon, target_vel, target_acc,
            action_recipe.aggressiveness, cars_size_margin, min_time=T_d)

        if vel_profile is not None:
            vel_profile = vel_profile.cut_by_time(BP_METRICS_TIME_HORIZON)
        return vel_profile

    def _calc_largest_safe_time(self, behavioral_state: SemanticBehavioralGridState, action: ActionRecipe,
                                vel_profile: VelocityProfile, ego_half_size: float, T_d: float) -> float:
        """
        For a lane change action, given ego velocity profile and behavioral_state, get two cars that may
        require faster lateral movement (the front overtaken car and the back interfered car) and calculate the last
        time, for which the safety holds w.r.t. these two cars.
        :param behavioral_state: semantic actions grid behavioral state
        :param action_lat_cell: either right, same or left
        :param vel_profile: the velocity profile of ego
        :param ego_half_size: half ego length
        :param T_d: time for comfortable lane change
        :return: the latest time, when ego is still safe
        """
        action_lat_cell = action.relative_lane
        ego_lon = behavioral_state.ego_state.road_localization.road_lon
        ego_time_delay1 = 0.8  # TODO: temp hack for two reasons: calm distance increasing if dist < 2 sec and overtaking of close car
        ego_time_delay2 = 1.6  # TODO: fix BUG of safety: it's unsafe for dyn action

        # check safety w.r.t. the front object on the target lane (if exists)
        if (action_lat_cell.value, SEMANTIC_CELL_LON_FRONT) in behavioral_state.road_occupancy_grid:
            forward_obj = behavioral_state.road_occupancy_grid[(action_lat_cell.value, SEMANTIC_CELL_LON_FRONT)]
            # smaller time_delay only in case of follow front car, to enable calm distance increasing if dist < 2 sec
            if action.action_type == ActionType.FOLLOW_VEHICLE and action_lat_cell == RelativeLane.SAME_LANE:
                time_delay = ego_time_delay1
            else:
                time_delay = ego_time_delay2
            forward_safe_time = ProfileSafety.calc_last_safe_time(ego_lon, ego_half_size, vel_profile, forward_obj[0],
                                                                  np.inf, time_delay)
            if forward_safe_time < vel_profile.total_time():
                obj_lon = forward_obj[0].road_localization.road_lon
                act_time = vel_profile.total_time()
                print('forward unsafe: act_type %d, rel_lat %d, dist=%.2f, act_time=%.2f final_dist=%.2f v_obj=%.2f '
                      'prof=(t=[%.2f %.2f %.2f] v=[%.2f %.2f %.2f], safe_time=%.2f)' %
                      (action.action_type.value, action_lat_cell.value, obj_lon - ego_lon,
                       act_time, obj_lon + act_time*forward_obj[0].v_x - (ego_lon+vel_profile.total_dist()),
                       forward_obj[0].v_x, vel_profile.t1, vel_profile.t2, vel_profile.t3, vel_profile.v_init,
                       vel_profile.v_mid, vel_profile.v_tar, forward_safe_time))
                return 0
        if action_lat_cell.value == SEMANTIC_CELL_LAT_SAME:  # continue on the same lane
            return np.inf  # don't check safety on other lanes

        # TODO: move it to a filter
        # check whether there is a car in the neighbor cell (same longitude)
        if (action_lat_cell.value, SEMANTIC_CELL_LON_SAME) in behavioral_state.road_occupancy_grid:
            print('side unsafe')
            return 0

        safe_time = np.inf
        # check safety w.r.t. the front object on the original lane (if exists)
        if (SEMANTIC_CELL_LAT_SAME, SEMANTIC_CELL_LON_FRONT) in behavioral_state.road_occupancy_grid:
            front_obj = behavioral_state.road_occupancy_grid[(SEMANTIC_CELL_LAT_SAME, SEMANTIC_CELL_LON_FRONT)]
            # check safety until half lateral time, since after that ego can escape laterally
            front_safe_time = ProfileSafety.calc_last_safe_time(ego_lon, ego_half_size, vel_profile, front_obj[0],
                                                                T_d/2, ego_time_delay1)
            if front_safe_time < np.inf:
                print('front_safe_time=%.2f front_dist=%.2f front_vel=%.2f' %
                      (front_safe_time, front_obj[0].road_localization.road_lon - ego_lon, front_obj[0].v_x))
            safe_time = min(safe_time, front_safe_time)

        # check safety w.r.t. the back object on the original lane (if exists)
        if (action_lat_cell.value, SEMANTIC_CELL_LON_REAR) in behavioral_state.road_occupancy_grid:
            back_obj = behavioral_state.road_occupancy_grid[(action_lat_cell.value, SEMANTIC_CELL_LON_REAR)]
            back_safe_time = ProfileSafety.calc_last_safe_time(ego_lon, ego_half_size, vel_profile, back_obj[0], T_d,
                                                               SAFE_DIST_TIME_DELAY)
            if back_safe_time < np.inf:
                print('back_safe_time=%.2f back_dist=%.2f back_vel=%.2f' %
                      (back_safe_time, ego_lon - back_obj[0].road_localization.road_lon, back_obj[0].v_x))
            safe_time = min(safe_time, back_safe_time)

        # print('front_time=%f back_time=%f forward_time=%f safe_time=%f' % \
        # (front_safe_time, back_safe_time, forward_safe_time, safe_time))
        return safe_time

    def _calc_action_cost(self, ego: EgoState, vel_profile: VelocityProfile, target_lane: int,
                          T_d: float, T_d_max: float, T_d_full: float, lat_dev: float, lane_width: float,
                          aggressiveness: AggressivenessLevel, goal: Optional[NavigationGoal]) -> [float, np.array]:
        """
        Calculate the cost of the action
        :param vel_profile: longitudinal velocity profile
        :param target_lane: target lane index
        :param T_d: lateral time according to the given aggressiveness level
        :param T_d_max: largest lateral time limited by safety
        :param T_d_full: lateral time of full lane change by calm action
        :param lat_dev: maximal lateral deviation from lane center (for half lane deviation, lat_dev = 1)
        :param aggressiveness: aggressiveness level of the action
        :param goal: navigation goal
        :return: the action's cost and the cost components array (for debugging)
        """
        vel_profile_time = vel_profile.total_time()
        efficiency_cost = comfort_cost = right_lane_cost = 0
        if vel_profile_time > 0:
            efficiency_cost = PlanEfficiencyMetric.calc_cost(vel_profile)
            comfort_cost = PlanComfortMetric.calc_cost(vel_profile, T_d, T_d_max, aggressiveness)
            right_lane_cost = PlanRightLaneMetric.calc_cost(vel_profile_time, target_lane)
        lane_deviation_cost = PlanLaneDeviationMetric.calc_cost(lat_dev)

        new_ego_localization = RoadLocalization(ego.road_localization.road_id, target_lane, target_lane*lane_width, 0,
                                                ego.road_localization.road_lon + vel_profile.total_dist(), 0)

        value_function = HeuristicStateActionRecipeEvaluator.value_function_approximation(
            new_ego_localization, vel_profile.v_tar, BP_METRICS_TIME_HORIZON - vel_profile_time, T_d_full, lane_width, goal)

        return efficiency_cost + comfort_cost + right_lane_cost + lane_deviation_cost + value_function, \
               np.array([efficiency_cost, comfort_cost, right_lane_cost, lane_deviation_cost, value_function])

    @staticmethod
    def value_function_approximation(road_localization: RoadLocalization, v_tar: float, time_period: float,
                                     T_d_full: float, lane_width: float, goal: Optional[NavigationGoal]) -> float:

        ego_road_id = road_localization.road_id
        ego_lane = road_localization.lane_num
        ego_lon = road_localization.road_lon
        rel_ego_lat = road_localization.intra_road_lat / lane_width
        empty_vel_profile = VelocityProfile(0, 0, 0, 0, 0, 0)

        time_period = max(0., time_period)
        efficiency_cost = right_lane_cost = future_lane_deviation_cost = future_comfort_cost = goal_cost = 0
        if time_period > 0:
            efficiency_cost = PlanEfficiencyMetric.calc_pointwise_cost_for_velocities(np.array([v_tar]))[0] * \
                              EFFICIENCY_COST_WEIGHT * time_period / WERLING_TIME_RESOLUTION
            right_lane_cost = ego_lane * time_period * RIGHT_LANE_COST_WEIGHT / WERLING_TIME_RESOLUTION

        if goal is None or ego_road_id != goal.road_id:  # no relevant goal
            if ego_lane > 0:  # distance in lanes from the rightest lane
                future_lane_deviation_cost = ego_lane * BP_METRICS_LANE_DEVIATION_COST_WEIGHT / WERLING_TIME_RESOLUTION
                future_comfort_cost = ego_lane * PlanComfortMetric.calc_cost(empty_vel_profile, T_d_full, T_d_full, AggressivenessLevel.CALM)
        elif ego_lane > goal.to_lane or ego_lane < goal.from_lane:  # outside of the lanes range of the goal
            if ego_lon >= goal.lon:  # we missed the goal
                goal_cost = BP_MISSING_GOAL_COST
            else:  # still did not arrive to the goal
                lanes_from_goal = max(rel_ego_lat - goal.to_lane, goal.from_lane - rel_ego_lat)
                T_d_max_per_lane = np.inf
                if v_tar * lanes_from_goal > 0:
                    T_d_max_per_lane = (goal.lon - ego_lon) / (v_tar * lanes_from_goal)  # required time for one lane change
                goal_comfort_cost = lanes_from_goal * PlanComfortMetric.calc_cost(
                    empty_vel_profile, T_d_full, T_d_max_per_lane, AggressivenessLevel.CALM)
                goal_cost = min(BP_MISSING_GOAL_COST, goal_comfort_cost)

        cost = efficiency_cost + future_comfort_cost + right_lane_cost + future_lane_deviation_cost + goal_cost
        return cost
