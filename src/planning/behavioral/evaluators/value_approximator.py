import numpy as np
from logging import Logger

from typing import Optional

from decision_making.src.global_constants import BP_METRICS_LANE_DEVIATION_COST_WEIGHT, BP_MISSING_GOAL_COST, \
    BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED, SAFE_DIST_TIME_DELAY, AGGRESSIVENESS_TO_LON_ACC, LON_ACC_LIMITS, \
    AV_TIME_DELAY, LON_ACC_TO_JERK_FACTOR, LON_JERK_COST_WEIGHT, VELOCITY_LIMITS, BP_RIGHT_LANE_COST_WEIGHT, \
    BP_EFFICIENCY_COST_WEIGHT
from decision_making.src.planning.behavioral.evaluators.cost_functions import BP_EfficiencyMetric, BP_ComfortMetric
from decision_making.src.planning.behavioral.evaluators.velocity_profile import VelocityProfile
from decision_making.src.planning.behavioral.data_objects import AggressivenessLevel, NavigationGoal, RelativeLane
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState, \
    RelativeLongitudinalPosition
from decision_making.src.planning.types import FS_SV, LIMIT_MIN, LIMIT_MAX, FS_SX
from decision_making.src.state.state import DynamicObject
from mapping.src.model.localization import RoadLocalization
from mapping.src.service.map_service import MapService


class ValueApproximator:
    def __init__(self, logger: Logger):
        self.logger = logger
        self.T_d_full = None
        self.calm_lat_comfort_cost = None

    def evaluate_state(self, behavioral_state: BehavioralGridState, goal: NavigationGoal) -> float:

        ego = behavioral_state.ego_state
        ego_loc = ego.road_localization
        ego_lane = ego_loc.lane_num
        ego_lon = ego_loc.road_lon
        ego_vel = ego.fstate[FS_SV]
        ego_length = ego.size.length
        empty_vel_profile = VelocityProfile(0, 0, 0, 0, 0, 0)

        lane_width = MapService.get_instance().get_road(ego.road_localization.road_id).lane_width
        if self.T_d_full is None:
            self.T_d_full, _, _ = VelocityProfile.calc_lateral_time(0, lane_width, lane_width, AggressivenessLevel.CALM)
            self.calm_lat_comfort_cost = BP_ComfortMetric.calc_cost(empty_vel_profile, self.T_d_full, self.T_d_full,
                                                           AggressivenessLevel.CALM, 0)

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

        num_lanes = MapService.get_instance().get_road(ego.road_localization.road_id).lanes_num
        dist_to_goal = goal.lon - ego_loc.road_lon  # TODO: use map & navigation

        # calculate cost for every lane change option
        forward_cost = ValueApproximator._calc_cost_for_lane(
            ego_vel, F.fstate[FS_SV], F.fstate[FS_SX] - ego_lon, ego_loc, RelativeLane.SAME_LANE,
            self.T_d_full, self.calm_lat_comfort_cost, (F.size.length + ego_length)/2, goal, dist_to_goal)

        right_cost = left_cost = np.inf
        if dist_to_goal >= self.T_d_full * ego_vel:  # enough time to change lane
            moderate_brake = -LON_ACC_LIMITS[LIMIT_MIN] / 2
            if ego_lane > 0 and R is None and VelocityProfile.is_safe_state(
                    ego_vel, RB.fstate[FS_SV], ego_lon - RB.fstate[FS_SX], SAFE_DIST_TIME_DELAY,
                            (RB.size.length + ego_length)/2, moderate_brake):
                right_cost = ValueApproximator._calc_cost_for_lane(
                    ego_vel, RF.fstate[FS_SV], RF.fstate[FS_SX] - ego_lon, ego_loc, RelativeLane.RIGHT_LANE,
                    self.T_d_full, self.calm_lat_comfort_cost, (RF.size.length + ego_length)/2, goal, dist_to_goal)

            if ego_lane < num_lanes - 1 and L is None and VelocityProfile.is_safe_state(
                    ego_vel, LB.fstate[FS_SV], ego_lon - LB.fstate[FS_SX], SAFE_DIST_TIME_DELAY,
                            (LB.size.length + ego_length)/2, moderate_brake):
                left_cost = ValueApproximator._calc_cost_for_lane(
                    ego_vel, LF.fstate[FS_SV], LF.fstate[FS_SX] - ego_lon, ego_loc, RelativeLane.LEFT_LANE,
                    self.T_d_full, self.calm_lat_comfort_cost, (LF.size.length + ego_length)/2, goal, dist_to_goal)

        min_cost = min(forward_cost, min(right_cost, left_cost))

        print('value: fwd=%.2f rgt=%.2f lft=%.2f' % (forward_cost, right_cost, left_cost))

        return min_cost

    @staticmethod
    def _calc_cost_for_lane(v_init: float, v_tar: float, cur_dist_from_obj: float, ego_loc: RoadLocalization,
                            lane_action: RelativeLane, T_d_full: float, calm_lat_comfort_cost: float,
                            cars_size_margin: float, goal: NavigationGoal, dist_to_goal: float) -> float:

        if lane_action != RelativeLane.SAME_LANE and \
                not VelocityProfile.is_safe_state(v_tar, v_init, cur_dist_from_obj, AV_TIME_DELAY, cars_size_margin):
            return np.inf  # unsafe lane change

        v_des = BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED
        final_lane = ego_loc.lane_num + lane_action.value
        eff_cost = comf_cost = non_right_lane_cost = goal_cost = 0

        if dist_to_goal > 0:
            final_dist_from_obj = v_tar * SAFE_DIST_TIME_DELAY + cars_size_margin
            dist_to_tar = cur_dist_from_obj - final_dist_from_obj  # may be negative since AV has lower reaction time
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
                time_tar_to_goal = (dist_to_goal - dist_to_tar) / max(0.001, v_tar)
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
                    time_to_goal = dist_to_goal / (0.5 * (v_init + v_des))  # approximated
                    if time_to_goal > time_tar_to_goal:  # target doesn't affect; FOLLOW_LANE to the goal
                        v_mid = v_init + acc * time_to_goal
                        vel_profile = VelocityProfile(v_init, time_to_goal, v_mid, 0, 0, v_init)
                    else:  # slow down to v_tar until the goal
                        time_to_goal = dist_to_goal / (0.5 * (v_init + v_tar))
                        vel_profile = VelocityProfile(v_init, time_to_goal, v_tar, 0, 0, v_tar)
                eff_cost = BP_EfficiencyMetric.calc_cost(vel_profile)
                comf_cost = BP_ComfortMetric.calc_cost(vel_profile, 0, np.inf, AggressivenessLevel.CALM, 0)

            non_right_lane_cost = final_lane * BP_RIGHT_LANE_COST_WEIGHT * vel_profile.total_time()
            if lane_action != RelativeLane.SAME_LANE:
                comf_cost += BP_METRICS_LANE_DEVIATION_COST_WEIGHT + calm_lat_comfort_cost

        if goal is not None and len(goal.lanes_list) > 0 and final_lane not in goal.lanes_list:
            goal_cost = BP_MISSING_GOAL_COST

        return eff_cost + comf_cost + non_right_lane_cost + goal_cost

    @staticmethod
    def _calc_goal_cost(ego_road_id: int, ego_lane: int, ego_lon: float, calm_lat_comfort_cost: float,
                        ego_vel: float, T_d_full: float, goal: NavigationGoal) -> float:

        goal_cost = 0
        if goal is None or ego_road_id != goal.road_id:  # no relevant goal
            if ego_lane > 0:  # distance in lanes from the rightest lane
                goal_cost = ego_lane * (BP_METRICS_LANE_DEVIATION_COST_WEIGHT + calm_lat_comfort_cost)
        elif len(goal.lanes_list) > 0 and ego_lane not in goal.lanes_list:  # outside of the lanes range of the goal
            if ego_lon < goal.lon:  # still did not arrive to the goal
                lanes_from_goal = np.min(np.abs(np.array(goal.lanes_list) - ego_lane))
                T_d_max_per_lane = np.inf
                # TODO: ego may break during moving to the goal, so T_d_max_per_lane may be greater
                if ego_vel * lanes_from_goal > 0:
                    T_d_max_per_lane = (goal.lon - ego_lon) / (ego_vel * lanes_from_goal)  # required time for one lane change
                empty_vel_profile = VelocityProfile(0, 0, 0, 0, 0, 0)
                lat_comfort_cost = BP_ComfortMetric.calc_cost(
                    empty_vel_profile, T_d_full, T_d_max_per_lane, AggressivenessLevel.CALM, 0)
                goal_cost = lanes_from_goal * (BP_METRICS_LANE_DEVIATION_COST_WEIGHT + lat_comfort_cost)
                goal_cost = min(BP_MISSING_GOAL_COST, goal_cost)
            else:  # we missed the goal
                goal_cost = BP_MISSING_GOAL_COST
        return goal_cost
