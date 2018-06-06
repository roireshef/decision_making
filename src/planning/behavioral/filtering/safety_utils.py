from logging import Logger
import numpy as np

from decision_making.src.global_constants import SAFETY_MARGIN_TIME_DELAY, SPECIFICATION_MARGIN_TIME_DELAY, \
    LON_ACC_LIMITS
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState, RelativeLane, \
    RelativeLongitudinalPosition
from decision_making.src.planning.behavioral.data_objects import ActionRecipe
from decision_making.src.planning.behavioral.evaluators.velocity_profile import VelocityProfile
from decision_making.src.planning.types import LIMIT_MIN


class SafetyUtils:

    def __init__(self):
        self.back_danger_lane = None
        self.back_danger_side = None
        self.back_danger_time = None
        self.front_blame = False

    def _calc_largest_safe_time(self, behavioral_state: BehavioralGridState, recipe: ActionRecipe, i: int,
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
        action_lat_cell = recipe.relative_lane
        ego = behavioral_state.ego_state
        ego_road = ego.road_localization
        ego_lon = ego_road.road_lon
        cur_time = ego.timestamp_in_sec

        forward_cell = (action_lat_cell, RelativeLongitudinalPosition.FRONT)
        front_cell = (RelativeLane.SAME_LANE, RelativeLongitudinalPosition.FRONT)
        side_rear_cell = (action_lat_cell, RelativeLongitudinalPosition.REAR)
        rear_cell = (RelativeLane.SAME_LANE, RelativeLongitudinalPosition.REAR)

        lane_change = (action_lat_cell != RelativeLane.SAME_LANE)
        lat_dist_to_target = abs(action_lat_cell.value - (ego_road.intra_lane_lat / lane_width - 0.5))  # in [0, 1.5]
        # increase time delay if ego does not move laterally according to the current action
        is_moving_laterally_to_target = (lane_change and ego_road.intra_road_yaw * action_lat_cell.value <= 0)

        # check safety w.r.t. the followed object on the target lane (if exists)
        if forward_cell in behavioral_state.road_occupancy_grid:
            followed_obj = behavioral_state.road_occupancy_grid[forward_cell][0].dynamic_object
            # calculate initial and final safety w.r.t. the followed object
            td = SAFETY_MARGIN_TIME_DELAY
            margin = 0.5 * (ego.size.length + followed_obj.size.length)
            (act_time, act_dist) = (vel_profile.total_time(), vel_profile.total_dist())
            obj_lon = followed_obj.road_localization.road_lon
            (end_ego_lon, end_obj_lon) = (ego_lon + act_dist, obj_lon + followed_obj.v_x * act_time)
            init_safe_dist = SafetyUtils.get_safety_dist(followed_obj.v_x, ego.v_x, obj_lon - ego_lon, td, margin)
            end_safe_dist = SafetyUtils.get_safety_dist(followed_obj.v_x, vel_profile.v_tar,
                                                            end_obj_lon - end_ego_lon, td, margin)

            # the action is unsafe if:  (change_lane and initially unsafe) or
            #                           (finally_unsafe and worse than initially)
            if (lane_change and (init_safe_dist <= 0 or end_safe_dist <= 0)) or \
                            end_safe_dist <= min(0., init_safe_dist):
                print('forward unsafe: %d(%d %d) rel_lat=%d dist=%.2f t=%.2f final_dist=%.2f v_obj=%.2f '
                      'prof=(t=[%.2f %.2f %.2f] v=[%.2f %.2f %.2f]) init_safe=%.2f final_safe=%.2f; td=%.2f' %
                      (i, recipe.action_type.value, recipe.aggressiveness.value, action_lat_cell.value,
                       obj_lon - ego_lon, act_time,
                       obj_lon + act_time * followed_obj.v_x - (ego_lon + act_dist),
                       followed_obj.v_x, vel_profile.t_first, vel_profile.t_flat, vel_profile.t_last, vel_profile.v_init,
                       vel_profile.v_mid, vel_profile.v_tar, init_safe_dist, end_safe_dist, td))
                return -1

        safe_time = np.inf
        if lane_change:  # for lane change actions check safety w.r.t. F, LB, RB
            # check whether there is a car in the neighbor cell (same longitude)
            if (action_lat_cell, RelativeLongitudinalPosition.PARALLEL) in behavioral_state.road_occupancy_grid:
                print('side unsafe rel_lat=%d: action %d' % (action_lat_cell.value, i))
                return -1

            # check safety w.r.t. the front object F on the original lane (if exists)
            if front_cell in behavioral_state.road_occupancy_grid:
                front_obj = behavioral_state.road_occupancy_grid[front_cell][0].dynamic_object
                # time delay decreases as function of lateral distance to the target: td_0 = td_T + 1
                # td_0 > td_T, since as latitude advances ego can escape laterally easier
                td_T = 0.  # dist to F after completing lane change.
                td_0 = SAFETY_MARGIN_TIME_DELAY * lat_dist_to_target
                # calculate last safe time w.r.t. F
                front_safe_time = SafetyUtils.calc_last_safe_time(vel_profile, ego_lon, ego_length,
                    front_obj.road_localization.road_lon, front_obj.v_x, front_obj.size.length, 0.75 * T_d, td_0, td_T)
                if front_safe_time < np.inf:
                    print('front_safe_time=%.2f action %d(%d %d): front_dist=%.2f front_vel=%.2f lat_d=%.2f td_0=%.2f td_T=%.2f' %
                          (front_safe_time, i, recipe.action_type.value, recipe.aggressiveness.value,
                           front_obj.road_localization.road_lon - ego_lon, front_obj.v_x, lat_dist_to_target, td_0, td_T))
                if front_safe_time <= 0:
                    return -1
                safe_time = min(safe_time, front_safe_time)

            # check safety w.r.t. the back object on the original lane (if exists)
            if side_rear_cell in behavioral_state.road_occupancy_grid:
                back_obj = behavioral_state.road_occupancy_grid[side_rear_cell][0].dynamic_object
                td = SPECIFICATION_MARGIN_TIME_DELAY
                # calculate last safe time w.r.t. LB or RB
                back_safe_time = SafetyUtils.calc_last_safe_time(vel_profile, ego_lon, ego_length,
                        back_obj.road_localization.road_lon, back_obj.v_x, back_obj.size.length, T_d, td)
                if back_safe_time < np.inf:
                    print('back_safe_time=%.2f action %d(%d %d): back_dist=%.2f back_vel=%.2f rel_lat=%.2f td=%.2f' %
                          (back_safe_time, i, recipe.action_type.value, recipe.aggressiveness.value,
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
                # if ego is on the danger_lane but still didn't reach the lane center,
                # and if this action is to the danger_lane center, then check safety w.r.t. the rear object
                if self.back_danger_lane == ego_road.lane_num and self.back_danger_side == action_lat_cell.value and \
                   ego_road.intra_lane_lat * action_lat_cell.value < 0 and rear_cell in behavioral_state.road_occupancy_grid:
                        td = SPECIFICATION_MARGIN_TIME_DELAY
                        rear_obj = behavioral_state.road_occupancy_grid[rear_cell][0].dynamic_object
                        # calculate last safe time w.r.t. R
                        back_safe_time = SafetyUtils.calc_last_safe_time(vel_profile, ego_lon, ego_length,
                                rear_obj.road_localization.road_lon, rear_obj.v_x, rear_obj.size.length, T_d, td)
                        if back_safe_time <= 0:
                            return -1
                        safe_time = min(safe_time, back_safe_time)
            else:  # after timeout, delete the danger flag
                self.back_danger_lane = None

        return safe_time

    @staticmethod
    def calc_last_safe_time(vel_profile: VelocityProfile, init_s_ego: float, ego_length: float, init_s_obj: float,
                            init_v_obj: float, obj_length: float, T: float, td_0: float, td_T: float=None) -> float:
        """
        Given ego velocity profile and dynamic object, calculate the last time, when the safety complies.
        :param vel_profile: piecewise linear velocity profile of ego
        :param init_s_ego: ego initial longitude
        :param ego_length: [m] length of ego
        :param init_s_obj: longitude of the dynamic object, for which the safety is tested
        :param init_v_obj: velocity of the dynamic object
        :param obj_length: length of the dynamic object
        :param T: maximal time to check the safety (usually T_d for safety w.r.t. F and LB)
        :param td_0: reaction time of the back car at time 0
        :param td_T: reaction time of the back car at time T_max (by default td_T = td_0)
        :return: the latest safe time
        """
        if T <= 0:
            return np.inf
        if td_T is None:
            td_T = td_0

        # first check if the profile is fully unsafe or fully safe
        margin = (ego_length + obj_length) / 2
        if init_s_ego < init_s_obj:
            v_front = init_v_obj
            v_back = vel_profile.v_init
            td_sign = -1
            front = 1  # whether the object is in front of ego
        else:
            v_back = init_v_obj
            v_front = vel_profile.v_init
            td_sign = 1
            front = 0
        back = 1 - front

        # if the initial state is unsafe, return -1
        if not SafetyUtils.is_safe_state(v_front, v_back, abs(init_s_obj - init_s_ego), td_0, margin):
            return -1

        # here the profile is PARTLY safe, find the latest safe time
        t, t_cum, s_ego, v_ego, a_ego = vel_profile.calc_profile_details(T)
        s_ego += init_s_ego
        td = td_0 + (td_T - td_0) * t_cum[:-1] / T  # time delay td changes from td_0 to td_T

        # calculate object's parameters at the beginning of each segment
        a_obj = np.repeat(0., t.shape[0])
        v_obj = np.repeat(init_v_obj, t.shape[0])
        # instead of moving ego by td, move the front/back object backward/forward by td
        s_obj = init_s_obj + init_v_obj * t_cum[:-1] + td_sign * v_obj * td

        # create (ego, obj) pairs of longitudes, velocities and accelerations for all segments
        (s, v, a) = (np.c_[s_ego[:-1], s_obj], np.c_[v_ego[:-1], v_obj], np.c_[a_ego, a_obj])

        # calculate last_safe_time
        last_safe_time = 0
        for seg in range(t.shape[0]):
            if t[seg] == 0:
                continue
            if a[seg, back] <= a[seg, front] and v[seg, back] <= v[seg, front]:
                last_safe_time += t[seg]
                continue
            seg_safe_time = SafetyUtils._calc_last_safe_time_for_segment(
                s[seg, front], v[seg, front], a[seg, front], s[seg, back], v[seg, back], a[seg, back], t[seg], margin)
            if seg_safe_time < t[seg]:
                # in case of front object decrease safe time by td, since seg_safe_time is the braking time
                safe_time = last_safe_time + max(0., seg_safe_time) - td[seg]*front
                return safe_time
            last_safe_time += t[seg]
        return np.inf  # always safe

    @staticmethod
    def get_safety_dist(v_front: float, v_back: float, dist: float, time_delay: float, margin: float,
                        max_brake: float=-LON_ACC_LIMITS[LIMIT_MIN]) -> float:
        """
        Calculate difference between the actual distance and minimal safe distance (longitudinal RSS formula)
        :param v_front: [m/s] front vehicle velocity
        :param v_back: [m/s] back vehicle velocity
        :param dist: [m] distance between the vehicles
        :param time_delay: time delay of the back vehicle
        :param margin: [m] cars sizes margin
        :param max_brake: [m/s^2] maximal deceleration of the vehicles
        :return: Positive if the back vehicle is safe, negative otherwise
        """
        safe_dist = max(0., v_back**2 - v_front**2) / (2*max_brake) + v_back*time_delay + margin
        return dist - safe_dist

    @staticmethod
    def is_safe_state(v_front: float, v_back: float, dist: float, time_delay: float, margin: float,
                      max_brake: float=-LON_ACC_LIMITS[LIMIT_MIN]) -> bool:
        """
        safety test by the longitudinal RSS formula
        :param v_front: [m/s] front vehicle velocity
        :param v_back: [m/s] back vehicle velocity
        :param dist: [m] distance between the vehicles
        :param time_delay: time delay of the back vehicle
        :param margin: [m] cars sizes margin
        :param max_brake: [m/s^2] maximal deceleration of the vehicles
        :return: True if the back vehicle is safe
        """
        return SafetyUtils.get_safety_dist(v_front, v_back, dist, time_delay, margin, max_brake) > 0

    @staticmethod
    def _calc_last_safe_time_for_segment(s_front: float, v_front: float, a_front: float, s_back: float, v_back: float,
                                         a_back: float, T: float, margin: float) -> float:
        """
        Given two vehicles with constant acceleration in time period [0, T], calculate the largest 0 <= t <= T,
        for which the second car (rear) is safe w.r.t. the first car in [0, t].
        :param s_front: first car longitude
        :param v_front: first car initial velocity
        :param a_front: first car acceleration
        :param s_back: second car longitude
        :param v_back: second car initial velocity
        :param a_back: second car acceleration
        :param T: time period
        :param margin: size margin of the cars
        :return: the largest safe time; if unsafe for t=0, return -1
        """
        if T < 0:
            return -1
        # the first vehicle is in front of the second one
        # s1(t) = s1 + v1*t + a1*t^2/2, s2(t) = s2 + v2*t + a2*t^2/2
        # v1(t) = v1 + a1*t, v2(t) = v2 + a2*t
        # safe_dist(t) = (v2(t)^2 - v1(t)^2) / (2*a_max) + margin
        # solve quadratic inequality: s1(t) - s2(t) - safe_dist(t) = A*t^2 + B*t + C >= 0

        a_max = -LON_ACC_LIMITS[LIMIT_MIN]

        C = s_front - s_back + (v_front * v_front - v_back * v_back) / (2 * a_max) - margin

        if C < 0:
            return -1  # the current state (for t=0) is not safe
        if T == 0:
            return 0  # the current state (for t=0) is safe

        A = (a_front - a_back) * (a_max + a_front + a_back) / (2 * a_max)
        B = (a_front * v_front - a_back * v_back) / a_max + v_front - v_back
        if A == 0 and B == 0:  # constant function C > 0
            return T  # for all t it's safe
        if A == 0:  # B != 0; linear inequality
            t = -C/B
            if t >= 0:
                return min(T, t)
            return T  # for all t it's safe

        # solve quadratic inequality
        discriminant = B*B - 4*A*C
        if discriminant < 0:
            return T  # for all t it's safe
        sqrt_disc = np.sqrt(discriminant)
        t_root1 = (-B - sqrt_disc)/(2*A)
        t_root2 = (-B + sqrt_disc)/(2*A)
        if t_root1 >= 0:
            return min(t_root1, T)
        if t_root2 >= 0:
            return min(t_root2, T)
        return T
