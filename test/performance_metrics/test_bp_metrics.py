from logging import Logger
from unittest.mock import patch
import pytest

import numpy as np

from decision_making.src.global_constants import BP_METRICS_TIME_HORIZON, BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED
from decision_making.src.planning.behavioral.architecture.components.evaluators.value_approximator import \
    ValueApproximator
from decision_making.src.planning.behavioral.architecture.data_objects import ActionType
from decision_making.src.planning.performance_metrics.behavioral.cost_functions import PlanRightLaneMetric, \
    PlanLaneDeviationMetric, PlanEfficiencyMetric, PlanComfortMetric
from decision_making.src.planning.performance_metrics.behavioral.velocity_profile import VelocityProfile, ProfileSafety
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.state.state import DynamicObject, ObjectSize


def test_behavioralScenarios_moveToLeftAndReturnToRight():
    des_vel = BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED
    slow_obj_vel = des_vel-2
    lane_width = 3.6
    cars_size_margin = 4
    logger = Logger("test_BP_metrics")
    value_approximator = ValueApproximator(logger)
    comfort_lane_change_time = VelocityProfile.calc_comfort_lateral_time(0, lane_width)

    road_points = np.array([np.arange(0, 1000), np.full(1000, 0)]).transpose()
    road_frenet = FrenetSerret2DFrame(road_points)

    # Test lane change to the left if front car's velocity is des_vel-2.
    # Test different initial velocities of ego, different distances from the slow car, empty/occupied left lane.
    # Desired result: for all initial velocities, don't overtake when dist=4 sec and overtake when dist=3 sec.

    for init_vel in [slow_obj_vel, des_vel]:
        for dist_from_slow_car_in_sec in [4, 2.5]:
            for left_action_type in [ActionType.FOLLOW_LANE, ActionType.FOLLOW_VEHICLE]:
                right_obj_lon = dist_from_slow_car_in_sec * init_vel
                right__vel_profile = VelocityProfile.calc_velocity_profile(
                    ActionType.FOLLOW_VEHICLE, 0, init_vel, right_obj_lon, slow_obj_vel, 0, 0, cars_size_margin, 0)

                if left_action_type == ActionType.FOLLOW_LANE:
                    min_time = BP_METRICS_TIME_HORIZON
                    left_obj_lon = None
                else:
                    min_time = comfort_lane_change_time
                    left_obj_lon = dist_from_slow_car_in_sec * init_vel  # the same longitude as the right car
                left__vel_profile = VelocityProfile.calc_velocity_profile(
                    left_action_type, 0, init_vel, left_obj_lon, des_vel, 0, 0, cars_size_margin, min_time)

                right__efficiency_cost = PlanEfficiencyMetric.calc_cost(right__vel_profile)
                right__comfort_cost = PlanComfortMetric.calc_cost(right__vel_profile, comfort_lane_change_time, np.inf)
                right__value = value_approximator.evaluate_state(
                    BP_METRICS_TIME_HORIZON - right__vel_profile.total_time(), slow_obj_vel, 0, comfort_lane_change_time)
                right__cost = right__efficiency_cost + right__comfort_cost + right__value

                # calculate the latest safe time
                ego_fpoint = np.array([0, lane_width/2])
                right_obj = DynamicObject(1, 0, right_obj_lon, lane_width/2, 0, 0, ObjectSize(cars_size_margin/2, 2, 0), 0, slow_obj_vel, 0, 0, 0)
                forward_safe_time = ProfileSafety.calc_last_safe_time(
                    ego_fpoint, cars_size_margin/2, left__vel_profile, right_obj, road_frenet, comfort_lane_change_time)

                left__efficiency_cost = PlanEfficiencyMetric.calc_cost(left__vel_profile)
                left__comfort_cost = PlanComfortMetric.calc_cost(left__vel_profile, comfort_lane_change_time, forward_safe_time)
                left__right_lane_cost = PlanRightLaneMetric.calc_cost(BP_METRICS_TIME_HORIZON, 1)
                left__lane_deviation_cost = PlanLaneDeviationMetric.calc_cost(comfort_lane_change_time)
                left__value = value_approximator.evaluate_state(
                    BP_METRICS_TIME_HORIZON - left__vel_profile.total_time(), des_vel, 1, comfort_lane_change_time)
                left__cost = left__efficiency_cost + left__comfort_cost + left__right_lane_cost + left__lane_deviation_cost + left__value
                if dist_from_slow_car_in_sec > 3:
                    assert right__cost < left__cost  # don't change lane
                else:
                    assert right__cost > left__cost  # change lane

    # test return to the right lane, when both lanes are empty
    left__vel_profile = VelocityProfile.calc_velocity_profile(
        ActionType.FOLLOW_LANE, 0, des_vel, None, des_vel, 0, 0, cars_size_margin, min_time=BP_METRICS_TIME_HORIZON)
    left__comfort_cost = PlanComfortMetric.calc_cost(left__vel_profile, comfort_lane_change_time, np.inf)
    left__right_lane_cost = PlanRightLaneMetric.calc_cost(BP_METRICS_TIME_HORIZON, 1)
    left__value = value_approximator.evaluate_state(
        BP_METRICS_TIME_HORIZON - left__vel_profile.total_time(), des_vel, 1, comfort_lane_change_time)
    left__cost = left__comfort_cost + left__right_lane_cost + left__value

    right__vel_profile = VelocityProfile.calc_velocity_profile(
        ActionType.FOLLOW_LANE, 0, des_vel, None, des_vel, 0, 0, cars_size_margin, min_time=BP_METRICS_TIME_HORIZON)
    right__comfort_cost = PlanComfortMetric.calc_cost(right__vel_profile, comfort_lane_change_time, np.inf)
    right__lane_deviation_cost = PlanLaneDeviationMetric.calc_cost(comfort_lane_change_time)
    right__value = value_approximator.evaluate_state(
        BP_METRICS_TIME_HORIZON - right__vel_profile.total_time(), des_vel, 0, comfort_lane_change_time)
    right__cost = right__comfort_cost + right__lane_deviation_cost + right__value
    assert right__cost < left__cost

    # test return to the right lane, when the right lane has a slower car
    init_vel = des_vel
    for dist_from_slow_car_in_sec in [8, 4]:
        right_obj_lon = dist_from_slow_car_in_sec * init_vel
        right__vel_profile = VelocityProfile.calc_velocity_profile(
            ActionType.FOLLOW_VEHICLE, 0, init_vel, right_obj_lon, slow_obj_vel, 0, 0, cars_size_margin,
            min_time=comfort_lane_change_time)

        right__efficiency_cost = PlanEfficiencyMetric.calc_cost(right__vel_profile)
        right__comfort_cost = PlanComfortMetric.calc_cost(right__vel_profile, comfort_lane_change_time, np.inf)
        right__lane_deviation_cost = PlanLaneDeviationMetric.calc_cost(comfort_lane_change_time)
        right__value = value_approximator.evaluate_state(
            BP_METRICS_TIME_HORIZON - right__vel_profile.total_time(), slow_obj_vel, 0, comfort_lane_change_time)
        right__cost = right__efficiency_cost + right__comfort_cost + right__lane_deviation_cost + right__value
        if dist_from_slow_car_in_sec > 6:
            assert right__cost < left__cost  # change lane to right
        else:
            assert right__cost > left__cost  # stay in the left lane
