import math
import numpy as np

from decision_making.src.global_constants import FORWARD_LOS_MAX_RANGE, CAR_DILATION_WIDTH, \
    HORIZONTAL_LOS_MAX_RANGE, SENSOR_OFFSET_FROM_FRONT, BEHAVIROAL_PLANNING_LOOKAHEAD_DISTANCE, MODERATE_DECELERATION, \
    TRAJECTORY_PLANNING_LOOKAHEAD_DISTANCE, LARGEST_CURVE_RADIUS, SAFETY_MIN_LOOKAHEAD_DIST


def is_in_front_of_ego(object_lat: float, object_width: float, ego_lat: float, ego_width: float,
                       obj_long_from_front_of_ego: float):
    """
    query that checks whether an object is in front of ego and in its lane.
    :param object_lat: latitude of object relative to the left road edge
    :param object_width: currently using constant width and ignoring yaw
    :param ego_lat: latitude of ego relative to left road edge
    :param ego_width: our car's width
    :param obj_long_from_front_of_ego: longitudal distance of object from from of our car
    :return: boolean
    """
    return math.fabs(object_lat - ego_lat) < (object_width + ego_width) / 2.0 and obj_long_from_front_of_ego > 0.0


def is_in_front_of_ego_not_in_same_lane(object_lat: float, object_width: float, ego_lat: float, ego_width: float,
                                        obj_long_from_front_of_ego: float):
    """
    query that checks whether an object is in front of ego and not in its lane.
    :param object_lat: latitude of object relative to the left road edge
    :param object_width: currently using constant width and ignoring yaw
    :param ego_lat: latitude of ego relative to left road edge
    :param ego_width: our car's width
    :param obj_long_from_front_of_ego: longitudal distance of object from from of our car
    :return: boolean
    """
    return math.fabs(object_lat - ego_lat) >= (object_width + ego_width) / 2.0 and obj_long_from_front_of_ego > 0.0


def calc_forward_sight_distance(static_objects: list, ego_lat: float, dyn_objects=None, min_speed_for_following=1.0):
    """
    Calculating the minimal distance of something that is in my lane
    :param static_objects: list of static objects, each is a dictionary
    :param ego_lat: latitude of ego relative to left road edge
    :param dyn_objects: TODO not used in July Milestone
    :param min_speed_for_following: TODO not used in July Milestone
    :return: float - minimal distance of something in my lane. If nothing is there, returns FORWARD_LOS_MAX_RANGE from
    Constants
    """
    min_static_object_long = FORWARD_LOS_MAX_RANGE
    for static_obj in static_objects:
        obj_lat = static_obj['full_lat']
        obj_width = static_obj['width']
        obj_lon = static_obj['relative_lon']
        if is_in_front_of_ego(obj_lat, obj_width, ego_lat, CAR_DILATION_WIDTH, obj_lon - SENSOR_OFFSET_FROM_FRONT):
            if obj_lon < min_static_object_long:
                min_static_object_long = obj_lon
    return min_static_object_long
    # TODO removing dynamic object handling for July milestone.
    # for (id, lat, lon, x, y, lat_velocity, long_velocity) in dyn_objects:
    #     lon -= sensor_offset_from_front
    #     if 0 - sensor_offset_from_front < lon < min_distance_in_my_lane:
    #         if is_in_front_of_ego(lat) and long_velocity <= min_speed_for_following:
    #             # ahead of me, and travelling less than 0.5 m/s in the long direction
    #             min_distance_in_my_lane = lon
    #             result = (x, y, min_distance_in_my_lane, 4, id)
    #             # 4- dynamic object in my lane ahead of me, and travelling less than 0.5 m/s in the long direction
    # return result


def calc_horizontal_sight_distance(static_objects: list, ego_lat: float, ego_vel: float,
                                   set_safety_lookahead_dist_by_ego_vel: bool = False):
    """
    calculates the minimal horizontal distance of static objects that are within a certain range tbd by
    set_safety_lookahead_dist_by_ego_vel
    :param static_objects: list of static objects, each is a dictionary
    :param ego_lat: latitude of ego relative to left road edge
    :param ego_vel: float - velocity of our car        return (min_safe_speed, constraintName)  # in meter/sec
    :param set_safety_lookahead_dist_by_ego_vel: parameter determining whether we use the trajectory length or the
    current breaking distance as the lookahead range
    :return: float - minimal distance of something in my lane. If nothing is there, returns HORIZONTAL_LOS_MAX_RANGE
    from Constants
    """

    min_horizontal_distance = HORIZONTAL_LOS_MAX_RANGE
    if set_safety_lookahead_dist_by_ego_vel:
        lookahead_distance = min(BEHAVIROAL_PLANNING_LOOKAHEAD_DISTANCE,
                                 (ego_vel * ego_vel / (2.0 * MODERATE_DECELERATION)))
        lookahead_distance = max(lookahead_distance, SAFETY_MIN_LOOKAHEAD_DIST)
    else:
        lookahead_distance = TRAJECTORY_PLANNING_LOOKAHEAD_DISTANCE

    for static_obj in static_objects:
        obj_lat = static_obj['full_lat']
        obj_width = static_obj['width']
        obj_lon = static_obj['relative_lon']
        obj_long_from_front_of_ego = obj_lon - SENSOR_OFFSET_FROM_FRONT
        if obj_long_from_front_of_ego <= lookahead_distance and \
                is_in_front_of_ego_not_in_same_lane(obj_lat, obj_width, ego_lat, CAR_DILATION_WIDTH,
                                                    obj_long_from_front_of_ego):
            relative_lat = math.fabs(obj_lat - ego_lat) - (obj_width + CAR_DILATION_WIDTH) / 2.0
            if relative_lat < min_horizontal_distance:
                min_horizontal_distance = relative_lat
    return min_horizontal_distance


def calc_road_turn_radius(path_points: np.ndarray):
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


    ########################################
    # NOT USED FOR JULY MILESTONE
    ########################################

    # in non-strict mode, we assume that only one pedestrian may exit to the road (like in ACDA). Strict mode has no such limitations
    # TODO - not used for July milestone
    # def CalcMaxSpeedForHorizontalOcclusions(occlusions_list, dyn_objects, sensor_distance_from_car_front, expected_car_vel, expected_pedestrian_vel, decel, tprt, strict_mode=False, log_file=None):
    #     VERBOSE = log_file is not None
    #     # collect all objects on my lane in order to filter them from horizontal LOS
    #     in_my_lane_list = []
    #     for (lat, _, _, _, _, _, obj_id) in occlusions_list:
    #         if obj_id not in in_my_lane_list and is_in_my_lane(lat):
    #             in_my_lane_list.append(obj_id)
    #
    #     occlusions = []
    #     for (lat, lon, x, y, octype, ang, obj_id) in occlusions_list:
    #         lon -= sensor_distance_from_car_front
    #         if lon > 0 and not(is_in_my_lane(lat)) and obj_id not in in_my_lane_list and octype < 3:
    #             occlusions.append((lat, lon, x, y, octype, ang, obj_id))
    #
    #     # adding dynamic objects to occlusions
    #     for (id, lat, lon, x, y, lat_velocity, long_velocity) in dyn_objects:
    #         lon -= sensor_distance_from_car_front
    #         if lat * lat_velocity < 0 < lon and not(is_in_my_lane(lat)):
    #             occlusions.append((lat, lon, x, y, 4, lat_velocity, id))  # adding the dyn objects as type 4
    #
    #     if len(occlusions) == 0:
    #         return (1000, -1, -1)
    #
    #     occlusions.sort(key=lambda tup: tup[1])  # sort by longitude
    #
    #     if VERBOSE:
    #         with open(log_file, 'a') as outfile:
    #             outfile.write('occlusions: ' + str(occlusions) + "\n\n")
    #
    #     # for each occlusion point calculate approaching time of a pedestrian/car to the lane according to octype
    #     times = []
    #     for (lat, lon, x, y, octype, ang_or_lat_velocity, _) in occlusions:  # ang_or_lat_velocity - ang if octype != 4 (not used). lat_vel otherwise (used)
    #         if octype == 3:
    #             vel = expected_car_vel
    #         elif octype == 4:  # dynamic object- take the current speed
    #             vel = math.fabs(ang_or_lat_velocity)
    #         else:
    #             vel = expected_pedestrian_vel
    #         times.append(math.fabs(lat) / vel)  # lat/vel = time for arriving to the road
    #
    #     # with open(self.log_file, 'a') as outfile:
    #     #     outfile.write('in horizontal occlusions calculation: \n')
    #     #     for i, occ in enumerate(occlusions):
    #     #         outfile.write('Dynamic Occ:  index: ' + str(i) + "occ: " + str(occ) + ', time: ' + str(times[i]) + " type: " + str(occ[4]) + '\n')
    #
    #     # for each point calculate max brake velocity and min pass velocity
    #     pass_vel_list = []
    #     brake_vel_list = []
    #     for i in range(len(occlusions)):
    #         lon = occlusions[i][1]
    #         t = times[i]
    #
    #         if not strict_mode:
    #             pass_vel_list.append(lon / (t + 1e-4))  # car's pass velocity = longitude/time
    #             max_brake_vel = math.sqrt(2*decel*lon + (decel*tprt)**2) - decel*tprt
    #             brake_vel_list.append(max_brake_vel)
    #         else:
    #             # THIS IS THE STRICTER METHOD FOR HLOS
    #             decel_t = max(0, t - tprt)
    #             # I safe if I can arrive to lon during braking before the pedestrian
    #             min_pass_vel = (lon + 0.5*decel*decel_t*decel_t) / (t + 1e-4)
    #             pass_vel_list.append(min_pass_vel)
    #             # I safe also if I can stop before lon
    #             max_brake_vel = math.sqrt(2*decel*lon + (decel*tprt)**2) - decel*tprt
    #             # I safe also if I can stop before t
    #             max_brake_vel = max(max_brake_vel, decel * decel_t)
    #             brake_vel_list.append(max_brake_vel)
    #
    #     if VERBOSE:
    #         with open(log_file, 'a') as outfile:
    #             outfile.write('brake_vel_list: ' + str(brake_vel_list) + "\n")
    #             outfile.write('pass_vel_list: ' + str(pass_vel_list) + '\n\n')
    #
    #     # for each i calculate min for brake_vel in subsequence i:n, and max for pass_vel in 0:i
    #     min_brake_vel_for_sublist = [min(brake_vel_list[i:]) for i in range(len(occlusions))]
    #     max_pass_vel_for_sublist = [max(pass_vel_list[0:i+1]) for i in range(len(occlusions))]
    #
    #     if VERBOSE:
    #         with open(log_file, 'a') as outfile:
    #             outfile.write('min_brake_vel_for_sublist: ' + str(min_brake_vel_for_sublist) + "\n")
    #             outfile.write('max_pass_vel_for_sublist: ' + str(max_pass_vel_for_sublist) + "\n")
    #             outfile.write('Done - horizontal occlusions calculation. \n\n')
    #
    #     # collect all safe intervals for which max_pass_vel[0:i] < min_brake_vel[i+1:n]
    #     safe_velocities = [(0, min_brake_vel_for_sublist[0], None, (occlusions[0][1], times[0], occlusions[0][4], occlusions[0][6]))]  # for the case when all points are far from my_loc
    #     for i in range(len(occlusions) - 1):
    #         if max_pass_vel_for_sublist[i] < min_brake_vel_for_sublist[i+1]:
    #             cur_dt = (occlusions[i][1], times[i], occlusions[i][4], occlusions[i][6])  # (lon, time, type, id)
    #             next_dt = (occlusions[i+1][1], times[i+1], occlusions[i+1][4], occlusions[i+1][6])  # (lon, time, type, id)
    #             if max_pass_vel_for_sublist[i] < safe_velocities[-1][1]:  # extend existing range
    #                 safe_velocities[-1] = (safe_velocities[-1][0], min_brake_vel_for_sublist[i+1], safe_velocities[-1][2], next_dt)
    #             else:
    #                 safe_velocities.append((max_pass_vel_for_sublist[i], min_brake_vel_for_sublist[i+1], cur_dt, next_dt))  # add new range
    #     if max_pass_vel_for_sublist[-1] < 1000:  # for the case when all points are close to my_loc
    #         if max_pass_vel_for_sublist[-1] < safe_velocities[-1][1]:  # extend existing range
    #             safe_velocities[-1] = (safe_velocities[-1][0], 1000, safe_velocities[-1][2], None)
    #         else:
    #             safe_velocities.append((max_pass_vel_for_sublist[-1], 1000, (occlusions[-1][1], times[-1], occlusions[-1][4], occlusions[-1][6]), None))
    #
    #     # return safe_velocities
    #     if safe_velocities[0][3] is None:
    #         return (1000, -1, -1)
    #     # TODO - vlad - make simpler by removing unnecessary data
    #     return (safe_velocities[0][1], safe_velocities[0][3][2], safe_velocities[0][3][3])  # (vel, type, obj_id)




    # This method returns the minimal distance from a vehicle which we are following.
    # it catches cases where the vehicle is in our lane, and its longitudal speed is greater than MIN_SPEED_FOR_FOLLOWING.
    # returns: distance in meters from the front of the cruise car.
    # TODO - not used for July milestone
    # def CalcFollowingDistance(dyn_objects, sensor_offset_from_front, min_speed_for_following):
    #     minimal_following_distance = float("inf")
    #     result = (None, None, minimal_following_distance, -1)
    #     for (id, lat, lon, x, y, lat_velocity, long_velocity) in dyn_objects:
    #         lon -= sensor_offset_from_front
    #         if is_in_my_lane(lat) and lon > 0 and lon < minimal_following_distance and long_velocity > min_speed_for_following:   #in my lane and ahead of me, with long speed > 5
    #             minimal_following_distance = lon
    #             result = (x, y, minimal_following_distance, id)
    #     return result
