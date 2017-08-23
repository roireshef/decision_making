# This code is based on https://en.wikipedia.org/wiki/Assured_Clear_Distance_Ahead
import math
from abc import ABCMeta
from typing import List

import numpy as np

from decision_making.src.global_constants import ACDA_NAME_FOR_LOGGING
from decision_making.src.planning.utils.acda_constants import *

from decision_making.src.state.state import DynamicObject, EgoState
from rte.python.logger.AV_logger import AV_Logger


class AcdaApi:
    _logger = AV_Logger.get_logger(ACDA_NAME_FOR_LOGGING)

    @staticmethod
    def set_logger(logger: AV_Logger):
        AcdaApi._logger = logger

    @staticmethod
    def compute_acda(objects_on_road: List[DynamicObject], ego_state: EgoState,
                     lookahead_path: np.ndarray) -> float:
        set_safety_lookahead_dist_by_ego_vel = False

        # get min long of static objects in my lane
        min_static_object_long = AcdaApi.calc_forward_sight_distance(objects_on_road, ego_state)
        # compute safe speed for forward line of sight
        safe_speed_forward_los = AcdaApi.calc_safe_speed_forward_line_of_sight(min_static_object_long)

        min_horizontal_distance_in_trajectory_range = AcdaApi.calc_horizontal_sight_distance(objects_on_road,
                                                                                             ego_state,
                                                                                             set_safety_lookahead_dist_by_ego_vel)
        safe_speed_horizontal_los = AcdaApi.calc_safe_speed_horizontal_distance_original_acda(
            min_horizontal_distance_in_trajectory_range)

        # TODO - refactor curve radius.
        # get curve radius
        curve_radius = AcdaApi.calc_road_turn_radius(lookahead_path)
        safe_speed_curve_radius = AcdaApi.calc_safe_speed_critical_speed(curve_radius)

        safe_speed = min(safe_speed_forward_los, safe_speed_horizontal_los, safe_speed_curve_radius)
        if safe_speed < 0:
            AcdaApi._logger.warn("safe_speed < 0")
        safe_speed = max(safe_speed, 0)
        return safe_speed

    ##############################################################
    # Formulas for computing ACDA given relevant distances
    ##############################################################
    @staticmethod
    def calc_safe_speed_forward_line_of_sight(forward_sight_distance: float) -> float:
        """
        Calculate safe speed when an obstacle is located on the way of ego, such that ego will brake safely
        :param forward_sight_distance: the forward distance to the obstacle in meters
        :return: the maximal safe speed in meter/sec
        """
        mu_e_times_g = (MU + SIN_ROAD_INCLINE) * G
        if forward_sight_distance < 0:
            AcdaApi._logger.warn("forward_sight_distance < 0")
        safe_speed_forward_los = math.sqrt(
            (mu_e_times_g * TPRT) ** 2 + 2.0 * mu_e_times_g * max(forward_sight_distance,
                                                                  0)) - mu_e_times_g * TPRT
        return max(0.0, safe_speed_forward_los)

    @staticmethod
    def calc_safe_speed_following_distance(following_distance: float) -> float:
        """
        Calculate safe speed while following after another car ("2 seconds law")
        :param following_distance: the current distance from the followed car in meters
        :return: the maximal safe speed in meters/sec
        """
        safe_speed_following_distance = following_distance / TIME_GAP
        if safe_speed_following_distance < 0:
            AcdaApi._logger.warn("safe_speed_following_distance < 0")
        return max(0.0, safe_speed_following_distance)

    @staticmethod
    def calc_safe_speed_critical_speed(curve_radius: float) -> float:
        """
        Calculate safe speed while going on a curve road, such that the centrifugal acceleration is bounded
        :param curve_radius: current curve radius of the road in meters
        :return: the maximal safe speed in meters/sec
        """
        mu_e_times_g = (MU + SIN_ROAD_INCLINE) * G
        safe_speed_critical_speed_arg = mu_e_times_g * curve_radius / (1.0 - (MU * SIN_ROAD_INCLINE))
        if safe_speed_critical_speed_arg < 0:
            AcdaApi._logger.warn("safe_speed_critical_speed argument < 0")
        safe_speed_critical_speed = math.sqrt(safe_speed_critical_speed_arg)
        return max(0.0, safe_speed_critical_speed)

    @staticmethod
    def calc_safe_speed_horizontal_distance_original_acda(min_horizontal_distance: float) -> float:
        """
        Calculate safe speed when an obstacle is NOT located on the way of ego, but a pedestrian may emerge from
         behind the obstacle. AV has either to brake before the pedestrian arrives to the lane, or to pass before it.
         This function does not consider the longitudinal distance to the obstacle, but assumes the obstacle
         is like a wall along the road, and the pedestrian may emerge from any place of the wall.
        :param min_horizontal_distance: the lateral distance to the obstacle in meters
        :return: the maximal safe speed in meters/sec
        """
        safe_speed = 2.0 * G * (MU + SIN_ROAD_INCLINE) * (min_horizontal_distance / HIDDEN_PEDESTRIAN_VEL - TPRT)
        if safe_speed < 0:
            AcdaApi._logger.warn("horizontal distance safe speed < 0")
        return max(2.0, safe_speed)

    ##############################################################
    # Utils for computing the relevant distances given the state
    ##############################################################
    @staticmethod
    def is_in_ego_trajectory(obj_lat: float, obj_width: float, ego_width: float, lateral_safety_margin: float) -> (
            bool):
        """
        query that checks whether an object is in ego's trajectory. Returns true if lane numbers are equal, or
        if |relative_obj_latitude| >= |sum of half object's and ego's width + safety_margin|.
        Does not check whether object is in
        front of ego. This needs to be done separately.
        :param obj_lat: lat distance w.r.t. ego ignoring road structure
        :param obj_width: object's width in meters
        :param ego_lane: our car's lane
        :param ego_width: our car's width
        :return: boolean
        """
        # return obj_lane == ego_lane or (math.fabs(obj_lane - ego_lane) <= 1 and obj_lat < (obj_width + ego_width) / 2.0)

        object_horizontal_distance = math.fabs(obj_lat) - ((obj_width + ego_width) / 2.0)
        return object_horizontal_distance <= lateral_safety_margin

    @staticmethod
    def calc_forward_sight_distance(static_objects: List[DynamicObject], ego_state: EgoState,
                                    dyn_objects: List[DynamicObject] = None) -> float:
        """
        Calculating the minimal distance of something that is in my lane
        :param static_objects: list of static objects, each is EnrichedObjectState
        :param ego_state: our car's state. Type EnrichedEgoState
        :param dyn_objects: TODO not used in July Milestone
        :param min_speed_for_following: TODO not used in July Milestone
        :return: float - minimal distance of something in my lane. If nothing is there, returns FORWARD_LOS_MAX_RANGE from
        Constants
        """
        min_static_object_long = FORWARD_LOS_MAX_RANGE
        for static_obj in static_objects:

            relative_road_localization = static_obj.get_relative_road_localization(ego_road_localization=ego_state.road_localization, ego_navigation_plan=ego_navigation_plan,
                map_api=map_api, max_lookahead_dist=BEHAVIORAL_PLANNING_LOOKAHEAD_DISTANCE)

            obj_lon = relative_road_localization.
            , obj_lat

            obj_lon = obj_lon - SENSOR_OFFSET_FROM_FRONT

            if obj_lon > 0 and AcdaApi.is_in_ego_trajectory(obj_lat=obj_lat, obj_width=static_obj.size.width,
                                                            ego_width=ego_state.size.width,
                                                            lateral_safety_margin=LATERAL_MARGIN_FROM_OBJECTS):

                if obj_lon < min_static_object_long:
                    min_static_object_long = obj_lon
        return min_static_object_long

    @staticmethod
    def calc_horizontal_sight_distance(static_objects: List[DynamicObject], ego_state: EgoState,
                                       set_safety_lookahead_dist_by_ego_vel: bool = False) -> float:
        """
        calculates the minimal horizontal distance of static objects that are within a certain range tbd by
        set_safety_lookahead_dist_by_ego_vel
        :param static_objects: list of static objects, each is a dictionary
        :param ego_state: our car's state. Type EnrichedEgoState
        :param set_safety_lookahead_dist_by_ego_vel: parameter determining whether we use the trajectory length or the
        current breaking distance as the lookahead range
        :return: float - minimal horizontal distance. If nothing is there, returns HORIZONTAL_LOS_MAX_RANGE
        from Constants
        """

        min_horizontal_distance = HORIZONTAL_LOS_MAX_RANGE
        if set_safety_lookahead_dist_by_ego_vel:
            # TODO: explain 2.0 factor & add to constants
            lookahead_distance = min(BEHAVIORAL_PLANNING_LOOKAHEAD_DISTANCE,
                                     (ego_state.v_x ** 2 / (2.0 * MODERATE_DECELERATION)))
            lookahead_distance = max(lookahead_distance, SAFETY_MIN_LOOKAHEAD_DIST)
        else:
            lookahead_distance = TRAJECTORY_PLANNING_LOOKAHEAD_DISTANCE

        for static_obj in static_objects:
            obj_lat = static_obj.rel_road_localization.rel_lat
            obj_width = static_obj.size.width
            obj_lon = static_obj.rel_road_localization.rel_lon
            obj_lon = obj_lon - SENSOR_OFFSET_FROM_FRONT
            if obj_lon <= lookahead_distance and not AcdaApi.is_in_ego_trajectory(obj_lat=obj_lat, obj_width=obj_width,
                                                                                  ego_width=ego_state.size.width,
                                                                                  lateral_safety_margin=LATERAL_MARGIN_FROM_OBJECTS):

                horizonal_distance = math.fabs(obj_lat) - (obj_width + ego_state.size.width) / 2.0
                if horizonal_distance < min_horizontal_distance:
                    min_horizontal_distance = horizonal_distance
        return min_horizontal_distance

    @staticmethod
    def calc_road_turn_radius(path_points: np.ndarray) -> float:
        """
        calculates turn radius given path points.
        This algorithm solves the "circle" fitting problem using a linear least squares method. Based on the paper
        "Circle fitting by linear and nonlinear least squares" (Coope, I.D.).
        :param path_points: np array of size 2 X m. Assumed to be uniformly distributed.
        :return: float - road turn radius
        """

        matrix_of_points = np.transpose(path_points)  # mX2 matrix of the points
        m = matrix_of_points.shape[0]
        matrix_of_points_squared = np.multiply(matrix_of_points, matrix_of_points)  # squaring each element
        y = matrix_of_points_squared.sum(axis=1)  # summing each row
        x = np.concatenate((matrix_of_points, np.ones((m, 1))), axis=1)
        transpose_x = np.transpose(x)
        xt_x = np.dot(transpose_x, x)
        if np.linalg.matrix_rank(xt_x) < 3:
            return LARGEST_CURVE_RADIUS
        theta = np.dot(np.dot(np.linalg.inv(xt_x), transpose_x), y)
        circle_center = 0.5 * theta[0:2]  # taking the first two elements in theta
        circle_center_x = circle_center[0]
        circle_center_y = circle_center[1]
        circle_radius = np.sqrt(theta[2] + circle_center_x ** 2 + circle_center_y ** 2)
        return circle_radius
