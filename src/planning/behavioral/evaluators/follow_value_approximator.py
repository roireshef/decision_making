import numpy as np
from logging import Logger

from decision_making.src.global_constants import BP_MISSING_GOAL_COST, \
    BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED, SPECIFICATION_MARGIN_TIME_DELAY, AGGRESSIVENESS_TO_LON_ACC, \
    SAFETY_MARGIN_TIME_DELAY, BP_RIGHT_LANE_COST_WEIGHT, BP_CALM_LANE_CHANGE_TIME
from decision_making.src.planning.behavioral.evaluators.cost_functions import BP_EfficiencyCost, BP_ComfortCost
from decision_making.src.planning.behavioral.evaluators.value_approximator import ValueApproximator
from decision_making.src.planning.behavioral.evaluators.velocity_profile import VelocityProfile
from decision_making.src.planning.behavioral.data_objects import NavigationGoal, RelativeLane, ActionSpec
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState, \
    RelativeLongitudinalPosition
from decision_making.src.planning.types import FS_SV, LIMIT_MIN, FS_SX, FS_DX
from decision_making.src.planning.utils.map_utils import MapUtils
from decision_making.src.state.state import DynamicObject
from mapping.src.model.localization import RoadLocalization
from mapping.src.service.map_service import MapService


class FollowValueApproximator(ValueApproximator):
    def __init__(self, logger: Logger):
        super().__init__(logger)
        self.logger = logger
        self.calm_lat_comfort_cost = None
        self.log = False

    def approximate(self, behavioral_state: BehavioralGridState, goal: NavigationGoal) -> float:

        ego = behavioral_state.ego_state
        ego_loc = ego.road_localization
        (ego_lane, ego_lon, ego_length, road_id) = (ego_loc.lane_num, ego_loc.road_lon, ego.size.length, ego_loc.road_id)
        road_frenet = MapUtils.get_road_rhs_frenet(ego)
        ego_fstate = MapUtils.get_ego_road_localization(ego, road_frenet)

        t_log = 18.1
        self.log = (abs(ego.timestamp_in_sec - 9.63 - t_log) < 0.2 and i == 40) or \
                   (abs(ego.timestamp_in_sec - 9.63 - t_log) < 0.2 and i == 64)

        lane_width = MapService.get_instance().get_road(road_id).lane_width
        if self.calm_lat_comfort_cost is None:
            spec = ActionSpec(0, 0, 0, lane_width)
            _, self.calm_lat_comfort_cost = BP_CostFunctions.calc_comfort_cost(
                ego_fstate, spec, 0, BP_CALM_LANE_CHANGE_TIME)

        F = None
        front_cell = (RelativeLane.SAME_LANE, RelativeLongitudinalPosition.FRONT)
        if front_cell in behavioral_state.road_occupancy_grid:
            F = behavioral_state.road_occupancy_grid[front_cell][0].dynamic_object

        map_based_nav_plan = MapService.get_instance().get_road_based_navigation_plan(road_id)
        # goal.lon - ego_loc.road_lon
        dist_to_goal = MapService.get_instance().get_longitudinal_difference(
            road_id, ego_loc.road_lon, goal.road_id, goal.lon, map_based_nav_plan)

        if self.log:
            print('ego_lon=%.f goal.lon=%.2f dist_to_goal=%.2f' % (ego_lon, goal.lon, dist_to_goal))

        # calculate cost for every lane change option
        F_vel, F_lon, F_len = FollowValueApproximator._get_vel_lon_len(F, 1)
        cost = self._calc_cost_for_lane(ego.v_x, F_vel, F_lon - ego_lon, ego_loc,
                                        (F_len + ego_length)/2, goal, dist_to_goal, ego_fstate)

        print('t=%.2f value %3d: F_vel=%.2f F_lon=%.2f, cost=%.2f' % (ego.timestamp_in_sec, i, F_vel, F_lon, cost))

        return cost

    def _calc_cost_for_lane(self, v_init: float, v_tar: float, cur_dist_from_obj: float, ego_loc: RoadLocalization,
                            cars_size_margin: float, goal: NavigationGoal, dist_to_goal: float, ego_fstate: np.array) -> float:

        v_des = BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED
        eff_cost = comf_cost = non_right_lane_cost = goal_cost = 0

        if dist_to_goal > 0:
            final_dist_from_obj = v_tar * SPECIFICATION_MARGIN_TIME_DELAY + cars_size_margin
            # dist_to_tar may be negative since AV has lower reaction time or inf for static action
            dist_to_tar = cur_dist_from_obj - final_dist_from_obj
            short_time = 0.5
            if dist_to_tar < v_tar * short_time:  # ego reached the target object, just follow it
                v_follow = max(0.001, min(v_tar, v_des))
                time = dist_to_goal / v_follow
                # vel_profile = VelocityProfile(v_follow, 0, v_follow, time, 0, v_follow)
                spec = ActionSpec(time, v_follow, ego_fstate[FS_SX] + time*v_follow, ego_fstate[FS_DX])
                eff_cost = BP_CostFunctions.calc_efficiency_cost(ego_fstate, spec)
                comf_cost = 0
            else:  # did not reach the target, first try FOLLOW_LANE if the target is far enough
                sgn = np.sign(v_des - v_init)
                acc = sgn * AGGRESSIVENESS_TO_LON_ACC[0]
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

                spec = ActionSpec(vel_profile.total_time(), vel_profile.v_tar,
                                  ego_fstate[FS_SX] + vel_profile.total_dist(), ego_fstate[FS_DX])

                eff_cost = BP_CostFunctions.calc_efficiency_cost(ego_fstate, spec)
                comf_cost, _ = BP_CostFunctions.calc_comfort_cost(ego_fstate, spec, 0, 0)

                if self.log:
                    print('eff_cost=%.2f time_to_goal=%.2f vel_prof=%s' %
                          (eff_cost, vel_profile.total_time(), vel_profile.__dict__))

            non_right_lane_cost = ego_loc.lane_num * BP_RIGHT_LANE_COST_WEIGHT * spec.t

            if self.log:
                print('right_lane_cost=%.2f comf=%.2f' % (non_right_lane_cost, comf_cost))

        if goal is not None and len(goal.lanes_list) > 0 and ego_loc.lane_num not in goal.lanes_list:
            goal_cost = BP_MISSING_GOAL_COST

        return eff_cost + comf_cost + non_right_lane_cost + goal_cost

    @staticmethod
    def _get_vel_lon_len(obj: DynamicObject, sgn: int) -> [float, float, float]:
        if obj is None:
            return BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED, sgn * np.inf, 0
        else:
            return obj.v_x, obj.road_localization.road_lon, obj.size.length
