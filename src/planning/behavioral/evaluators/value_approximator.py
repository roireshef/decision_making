import numpy as np
from logging import Logger

from typing import Optional

from decision_making.src.global_constants import BP_METRICS_LANE_DEVIATION_COST_WEIGHT, BP_MISSING_GOAL_COST, \
    BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED, SAFE_DIST_TIME_DELAY, AGGRESSIVENESS_TO_LON_ACC, LON_ACC_LIMITS, \
    AV_TIME_DELAY, BP_RIGHT_LANE_COST_WEIGHT, BP_CALM_LANE_CHANGE_TIME
from decision_making.src.planning.behavioral.evaluators.cost_functions import BP_EfficiencyMetric, BP_ComfortMetric
from decision_making.src.planning.behavioral.evaluators.velocity_profile import VelocityProfile
from decision_making.src.planning.behavioral.data_objects import AggressivenessLevel, NavigationGoal, RelativeLane, \
    ActionSpec
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState, \
    RelativeLongitudinalPosition
from decision_making.src.planning.types import FS_SV, LIMIT_MIN, LIMIT_MAX, FS_SX, FS_DX
from decision_making.src.planning.utils.map_utils import MapUtils
from decision_making.src.state.state import DynamicObject, ObjectSize
from mapping.src.model.localization import RoadLocalization
from mapping.src.service.map_service import MapService


class ValueApproximator:
    def __init__(self, logger: Logger):
        self.logger = logger
        self.calm_lat_comfort_cost = None

    def approximate(self, behavioral_state: BehavioralGridState, goal: NavigationGoal) -> float:

        ego = behavioral_state.ego_state
        ego_loc = ego.road_localization
        (ego_lane, ego_lon, ego_length, road_id) = (ego_loc.lane_num, ego_loc.road_lon, ego.size.length, ego_loc.road_id)
        road_frenet = MapUtils.get_road_rhs_frenet(ego)
        ego_fstate = MapUtils.get_ego_road_localization(ego, road_frenet)

        lane_width = MapService.get_instance().get_road(road_id).lane_width
        if self.calm_lat_comfort_cost is None:
            spec = ActionSpec(0, 0, 0, lane_width)
            _, self.calm_lat_comfort_cost = BP_ComfortMetric.calc_cost(
                ego_fstate, spec, BP_CALM_LANE_CHANGE_TIME)

        F = LF = RF = LB = RB = L = R = None
        if (RelativeLane.SAME_LANE, RelativeLongitudinalPosition.FRONT) in behavioral_state.road_occupancy_grid:
            F = behavioral_state.road_occupancy_grid[(RelativeLane.SAME_LANE, RelativeLongitudinalPosition.FRONT)][0].dynamic_object
        if (RelativeLane.LEFT_LANE, RelativeLongitudinalPosition.FRONT) in behavioral_state.road_occupancy_grid:
            LF = behavioral_state.road_occupancy_grid[(RelativeLane.LEFT_LANE, RelativeLongitudinalPosition.FRONT)][0].dynamic_object
        if (RelativeLane.RIGHT_LANE, RelativeLongitudinalPosition.FRONT) in behavioral_state.road_occupancy_grid:
            RF = behavioral_state.road_occupancy_grid[(RelativeLane.RIGHT_LANE, RelativeLongitudinalPosition.FRONT)][0].dynamic_object
        if (RelativeLane.LEFT_LANE, RelativeLongitudinalPosition.REAR) in behavioral_state.road_occupancy_grid:
            LB = behavioral_state.road_occupancy_grid[(RelativeLane.LEFT_LANE, RelativeLongitudinalPosition.REAR)][0].dynamic_object
        if (RelativeLane.RIGHT_LANE, RelativeLongitudinalPosition.REAR) in behavioral_state.road_occupancy_grid:
            RB = behavioral_state.road_occupancy_grid[(RelativeLane.RIGHT_LANE, RelativeLongitudinalPosition.REAR)][0].dynamic_object
        if (RelativeLane.LEFT_LANE, RelativeLongitudinalPosition.PARALLEL) in behavioral_state.road_occupancy_grid:
            L = behavioral_state.road_occupancy_grid[(RelativeLane.LEFT_LANE, RelativeLongitudinalPosition.PARALLEL)][0].dynamic_object
        if (RelativeLane.RIGHT_LANE, RelativeLongitudinalPosition.PARALLEL) in behavioral_state.road_occupancy_grid:
            R = behavioral_state.road_occupancy_grid[(RelativeLane.RIGHT_LANE, RelativeLongitudinalPosition.PARALLEL)][0].dynamic_object

        num_lanes = MapService.get_instance().get_road(road_id).lanes_num
        map_based_nav_plan = MapService.get_instance().get_road_based_navigation_plan(road_id)
        # goal.lon - ego_loc.road_lon
        dist_to_goal = MapService.get_instance().get_longitudinal_difference(
            road_id, ego_loc.road_lon, goal.road_id, goal.lon, map_based_nav_plan)

        # calculate cost for every lane change option
        F_vel, F_lon, F_len = ValueApproximator._get_vel_lon_len(F, 1)
        forward_cost = self._calc_cost_for_lane(ego.v_x, F_vel, F_lon - ego_lon, ego_loc, RelativeLane.SAME_LANE,
                                                (F_len + ego_length)/2, goal, dist_to_goal, ego_fstate)

        right_cost = left_cost = np.inf
        lane_change_time = 5
        if dist_to_goal >= lane_change_time * ego.v_x:  # enough time to change lane
            moderate_brake = -LON_ACC_LIMITS[LIMIT_MIN] / 2
            RB_vel, RB_lon, RB_len = ValueApproximator._get_vel_lon_len(RB, -1)
            if ego_lane > 0 and R is None and VelocityProfile.is_safe_state(
                    ego.v_x, RB_vel, ego_lon - RB_lon, SAFE_DIST_TIME_DELAY, (RB_len + ego_length)/2, moderate_brake):
                RF_vel, RF_lon, RF_len = ValueApproximator._get_vel_lon_len(RF, 1)
                right_cost = self._calc_cost_for_lane(ego.v_x, RF_vel, RF_lon - ego_lon, ego_loc, RelativeLane.RIGHT_LANE,
                                                      (RF_len + ego_length)/2, goal, dist_to_goal, ego_fstate)

            LB_vel, LB_lon, LB_len = ValueApproximator._get_vel_lon_len(LB, -1)
            if ego_lane < num_lanes - 1 and L is None and VelocityProfile.is_safe_state(
                    ego.v_x, LB_vel, ego_lon - LB_lon, SAFE_DIST_TIME_DELAY, (LB_len + ego_length)/2, moderate_brake):
                LF_vel, LF_lon, LF_len = ValueApproximator._get_vel_lon_len(LF, 1)
                left_cost = self._calc_cost_for_lane(ego.v_x, LF_vel, LF_lon - ego_lon, ego_loc, RelativeLane.LEFT_LANE,
                                                     (LF_len + ego_length)/2, goal, dist_to_goal, ego_fstate)

        min_cost = min(forward_cost, min(right_cost, left_cost))

        print('value: fwd=%.2f rgt=%.2f lft=%.2f' % (forward_cost, right_cost, left_cost))

        return min_cost

    def _calc_cost_for_lane(self, v_init: float, v_tar: float, cur_dist_from_obj: float, ego_loc: RoadLocalization,
                            lane_action: RelativeLane, cars_size_margin: float, goal: NavigationGoal,
                            dist_to_goal: float, ego_fstate: np.array) -> float:

        if lane_action != RelativeLane.SAME_LANE and \
                not VelocityProfile.is_safe_state(v_tar, v_init, cur_dist_from_obj, AV_TIME_DELAY, cars_size_margin):
            return np.inf  # unsafe lane change

        v_des = BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED
        final_lane = ego_loc.lane_num + lane_action.value
        eff_cost = comf_cost = non_right_lane_cost = goal_cost = 0

        if dist_to_goal > 0:
            final_dist_from_obj = v_tar * SAFE_DIST_TIME_DELAY + cars_size_margin
            # dist_to_tar may be negative since AV has lower reaction time or inf for static action
            dist_to_tar = cur_dist_from_obj - final_dist_from_obj
            short_time = 0.5
            if dist_to_tar < v_tar * short_time:  # ego reached the target object, just follow it
                v_follow = max(0.001, min(v_tar, v_des))
                time = dist_to_goal / v_follow
                vel_profile = VelocityProfile(v_follow, 0, v_follow, time, 0, v_follow)
                eff_cost = BP_EfficiencyMetric.calc_cost(vel_profile)
                comf_cost = 0
            else:  # did not reach the target, first try FOLLOW_LANE if the target is far enough
                sgn = np.sign(v_des - v_init)
                acc = sgn * AGGRESSIVENESS_TO_LON_ACC[AggressivenessLevel.CALM.value]
                time_tar_to_goal = (dist_to_goal - dist_to_tar) / max(0.001, v_tar)  # -inf for static action
                t1 = (v_des - v_init) / acc
                dist_acc = 0.5 * (v_init + v_des) * t1
                if dist_to_goal > dist_acc:
                    t2_to_goal = (dist_to_goal - dist_acc) / v_des
                    time_to_goal = t1 + t2_to_goal
                    if time_to_goal > time_tar_to_goal:  # target arrives to goal before ego, then FOLLOW_LANE till the goal
                        v_mid = v_init + acc * t1
                        vel_profile = VelocityProfile(v_init, t1, v_mid, t2_to_goal, 0, v_mid)
                    else:  # ego may arrive to the goal before target, then follow target during time_tar_to_goal
                        vel_profile = VelocityProfile.calc_profile_given_T(v_init, time_tar_to_goal, dist_to_tar, v_tar)
                else:  # arrives to the goal during acceleration
                    t1 = dist_to_goal / (0.5 * (v_init + v_des))  # approximated
                    if t1 > time_tar_to_goal:  # target doesn't affect; FOLLOW_LANE to the goal
                        v_mid = v_init + acc * t1
                        vel_profile = VelocityProfile(v_init, t1, v_mid, 0, 0, v_mid)
                    else:  # target affects, FOLLOW_CAR
                        vel_profile = VelocityProfile.calc_profile_given_T(v_init, time_tar_to_goal, dist_to_tar, v_tar)
                eff_cost = BP_EfficiencyMetric.calc_cost(vel_profile)
                spec = ActionSpec(vel_profile.total_time(), vel_profile.v_tar,
                                  ego_fstate[FS_SX] + vel_profile.total_dist(), ego_fstate[FS_DX])
                comf_cost, _ = BP_ComfortMetric.calc_cost(ego_fstate, spec, 0)
                # comf_cost = BP_ComfortMetric.calc_cost(vel_profile, 0, np.inf, 0)

            non_right_lane_cost = final_lane * BP_RIGHT_LANE_COST_WEIGHT * vel_profile.total_time()
            if lane_action != RelativeLane.SAME_LANE:
                comf_cost += BP_METRICS_LANE_DEVIATION_COST_WEIGHT + self.calm_lat_comfort_cost

        if goal is not None and len(goal.lanes_list) > 0 and final_lane not in goal.lanes_list:
            goal_cost = BP_MISSING_GOAL_COST

        return eff_cost + comf_cost + non_right_lane_cost + goal_cost

    @staticmethod
    def _get_vel_lon_len(obj: DynamicObject, sgn: int) -> [float, float, float]:
        if obj is None:
            return BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED, sgn * np.inf, 0
        else:
            return obj.v_x, obj.road_localization.road_lon, obj.size.length
