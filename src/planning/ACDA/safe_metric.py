# This code is based on https://en.wikipedia.org/wiki/Assured_Clear_Distance_Ahead
import math

from decision_making.src.global_constants import G, SIN_ROAD_INCLINE, HIDDEN_PEDESTRIAN_VEL


class SafeMetric:
    def __init__(self, mu: float = 0.7, tprt: float = 0.5, time_gap: float = 2.0):
        self.mu = mu  # the friction coefficient (unitless), a function of the tire type and road conditions
        self.decel = mu * G
        self.e = SIN_ROAD_INCLINE
        self.tprt = tprt  # for our agent, it is the perception-reaction time in seconds. For human it is 0.72s.
        self.time_gap = time_gap  # this is the X (usually, 2) second rule. Unit is seconds.

    def calc_safe_speed_forward_line_of_sight(self, forward_sight_distance: float):  # forwardSightDistance is in meters
        """
        Calculate safe speed when an obstacle is located on the way of ego, such that ego will brake safely
        :param forward_sight_distance: the forward distance to the obstacle
        :return: the maximal safe speed in meter/sec
        """
        mu_e_times_g = (self.mu + self.e) * G
        safe_speed_forward_los = math.sqrt(
            (mu_e_times_g * self.tprt) ** 2 + 2.0 * mu_e_times_g * max(forward_sight_distance,
                                                                       0)) - mu_e_times_g * self.tprt
        return max(0.0, safe_speed_forward_los)  #

    def calc_safe_speed_following_distance(self, following_distance: float):
        """
        Calculate safe speed while following after another car ("2 seconds law")
        :param following_distance: the current distance from the followed car in meters
        :return: the maximal safe speed in meters/sec
        """
        safe_speed_following_distance = following_distance / self.time_gap
        return max(0.0, safe_speed_following_distance)

    def calc_safe_speed_critical_speed(self, curve_radius: float):
        """
        Calculate safe speed while going on a curve road, such that the centrifugal acceleration is bounded
        :param curve_radius: current curve radius of the road in meters
        :return: the maximal safe speed in meters/sec
        """
        mu_e_times_g = (self.mu + self.e) * G
        safe_speed_critical_speed = math.sqrt(
            mu_e_times_g * curve_radius / (1.0 - (self.mu * self.e)))
        return max(0.0, safe_speed_critical_speed)

    def calc_safe_speed_horizontal_distance_original_acda(self, min_horizontal_distance: float):
        """
        Calculate safe speed when an obstacle is NOT located on the way of ego, but a pedestrian may emerge from
         behind the obstacle. AV has either to brake before the pedestrian arrives to the lane, or to pass before it.
         This function does not consider the longitudinal distance to the obstacle, but assumes the obstacle
         is like a wall along the road, and the pedestrian may emerge from any place of the wall.
        :param min_horizontal_distance: the lateral distance to the obstacle in meters
        :return: the maximal safe speed in meters/sec
        """
        safe_speed = 2.0 * G * (self.mu + self.e) * (min_horizontal_distance / HIDDEN_PEDESTRIAN_VEL - self.tprt)
        return max(2.0, safe_speed)


        ########################################
        # NOT USED FOR JULY MILESTONE
        ########################################
        #
        # def calc_safe_speed(self, forward_sight_distance, horiz_safe_velocity, following_distance, curve_radius, safe_speed_surface_control, legal_speed_limit):
        #     # safeSpeedSurfaceControl and legalSpeedLimit are in meter/second. safeSpeedSurfaceControl is affected by several factors, for example, tire speed rating and many more
        #     safe_speed_forwrad_LOS = self.calc_safe_speed_forward_line_of_sight(forward_sight_distance)
        #
        #     safe_speed_horizontal_LOS = horiz_safe_velocity  # upper limit of first interval
        #
        #     safe_speed_following_distance = self.calc_safe_speed_following_distance(following_distance)
        #     safe_speed_critical_speed = self.calc_safe_speed_critical_speed(curve_radius)
        #     safe_speeds = {"forward LOS" : safe_speed_forwrad_LOS,
        #                 "horizontal LOS" : safe_speed_horizontal_LOS,
        #                 "following distance" : safe_speed_following_distance,
        #                 "critical curve speed" : safe_speed_critical_speed,
        #                 "surface control speed" : safe_speed_surface_control,
        #                 "legal speed limit" : legal_speed_limit}
        #
        #     min_safe_speed = min( safe_speeds.values() )
        #     minIndex = safe_speeds.values().index(min_safe_speed)
        #     constraintName = safe_speeds.keys()[minIndex]
        #     return (min_safe_speed, constraintName)  # in meter/sec


        # def CalcSafeMetric(self, safeSpeed, cruiseCarSpeed):
        #     if cruiseCarSpeed < safeSpeed:
        #         safeMetric = 0  # i.e., safe
        #     else:
        #         safeMetric =  ( cruiseCarSpeed - safeSpeed ) ** 2
        #     return safeMetric
