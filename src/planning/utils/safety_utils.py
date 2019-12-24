import numpy as np
from logging import Logger
from decision_making.src.global_constants import LON_ACC_LIMITS, EPS, TRAJECTORY_TIME_RESOLUTION
from decision_making.src.planning.types import FrenetTrajectories2D, FrenetTrajectory2D, FS_SX, FS_SV, FS_DX
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor

LON_SAFETY_ACCEL_DURING_RESPONSE = 0.
LON_SAFETY_BACK_ACTOR_MAX_DECEL = 3.8


class SafetyRSS:

    @staticmethod
    def get_lon_safe_dist(ego_trajectories: FrenetTrajectories2D, trajectory_lengths: np.array, ego_response_time: float,
                          obj_trajectory: FrenetTrajectory2D, obj_response_time: float,
                          margin: float, logger: Logger, ego_behind_max_brake: float = -LON_ACC_LIMITS[0],
                          ego_ahead_max_brake: float = LON_SAFETY_BACK_ACTOR_MAX_DECEL) -> np.array:
        """
        Calculate longitudinal safety between ego and another object for all timestamps.
        Longitudinal safety between two objects considers only their longitudinal data: longitude and longitudinal velocity.
        Longitudinal RSS formula considers distance reduction during the reaction time and difference between
        objects' braking distances.
        An object is defined safe if it's safe either longitudinally OR laterally.
        :param ego_trajectories: ego Frenet trajectories: 3D tensor of shape: traj_num x timestamps_num x 6
        :param trajectory_lengths: array of lengths of ego_trajectories
        :param ego_response_time: [sec] ego response time
        :param obj_trajectory: object's Frenet trajectory: 2D matrix: timestamps_num x 6
        :param obj_response_time: [sec] object's response time
        :param margin: [m] cars' lengths half sum
        :param logger:
        :param ego_behind_max_brake: [m/s^2] maximal deceleration of both objects for front actor
        :param ego_ahead_max_brake: [m/s^2] maximal deceleration of both objects for back actor
        :return: normalized longitudinal safety distance per timestamp. 2D matrix shape: traj_num x timestamps_num
        """
        # extract the relevant longitudinal data from the trajectories
        ego_lon, ego_vel, ego_lat = ego_trajectories[..., FS_SX], ego_trajectories[..., FS_SV], ego_trajectories[..., FS_DX]
        ego_trajectories_s = ego_trajectories[..., :FS_DX]
        if ego_trajectories.ndim == 2:  # single ego trajectory
            ego_lon = ego_lon[np.newaxis]
            ego_vel = ego_vel[np.newaxis]
            ego_lat = ego_lat[np.newaxis]
            ego_trajectories_s = ego_trajectories_s[np.newaxis]

        obj_lon, obj_vel, obj_lat = obj_trajectory[:, FS_SX], obj_trajectory[:, FS_SV], obj_trajectory[:, FS_DX]

        # find trajectories, for which ego is ahead the object, when they are closest laterally
        min_lat_dist_times = np.argmin(np.abs(ego_lat - obj_lat), axis=-1)
        ego_ahead = (ego_lon[np.arange(ego_lon.shape[0]), min_lat_dist_times] > obj_lon[min_lat_dist_times])
        ego_behind = np.logical_not(ego_ahead)

        marginal_safe_dist = np.zeros_like(ego_lon)

        if ego_ahead.any() > 0:
            ego_ahead_lon = ego_lon[ego_ahead]
            ego_ahead_vel = ego_vel[ego_ahead]
            for ego_s, traj_length in zip(ego_ahead_lon, trajectory_lengths):
                ego_s[traj_length:] = np.inf

            # The worst-case velocity of the rear object (either ego or another object) may increase during its reaction
            # time, since it may accelerate before it starts to brake.
            obj_vel_after_reaction_time = obj_vel + obj_response_time * LON_SAFETY_ACCEL_DURING_RESPONSE

            # longitudinal RSS formula considers distance reduction during the reaction time and difference between
            # objects' braking distances
            obj_acceleration_dist = 0.5 * LON_SAFETY_ACCEL_DURING_RESPONSE * obj_response_time ** 2
            min_safe_dist = np.maximum((obj_vel_after_reaction_time ** 2 - ego_ahead_vel ** 2) / (2 * ego_ahead_max_brake), 0) + \
                            (obj_vel * obj_response_time + obj_acceleration_dist) + margin
            marginal_safe_dist[ego_ahead] = ego_ahead_lon - obj_lon - min_safe_dist

        if ego_behind.any():
            ego_behind_lon = ego_lon[ego_behind]
            ego_behind_vel = ego_vel[ego_behind]
            for ego_s, traj_length in zip(ego_behind_lon, trajectory_lengths):
                ego_s[traj_length:] = -np.inf

            # extrapolate ego trajectories ego_response_time seconds beyond their end state
            dt = TRAJECTORY_TIME_RESOLUTION
            predictor = RoadFollowingPredictor(logger)
            extrapolated_times = np.arange(dt, ego_response_time + EPS, dt)
            last_ego_states_s = ego_trajectories_s[ego_behind, -1]
            ego_extrapolation = predictor.predict_1d_frenet_states(last_ego_states_s, extrapolated_times)

            ext_ego_lon = np.concatenate((ego_behind_lon, ego_extrapolation[..., FS_SX]), axis=1)
            ext_ego_vel = np.concatenate((ego_behind_vel, ego_extrapolation[..., FS_SV]), axis=1)

            # we assume ego continues its trajectory during its reaction time, so we compute the difference between
            # object's braking distance from any moment and delayed braking distance of ego
            delay_shift = ego_extrapolation.shape[1]
            braking_distances_diff = (ext_ego_vel[:, delay_shift:] ** 2 - obj_vel ** 2) / (2 * ego_behind_max_brake)
            marginal_safe_dist[ego_behind] = obj_lon - ext_ego_lon[:, delay_shift:] - braking_distances_diff - margin

        return marginal_safe_dist if ego_trajectories.ndim > 2 else marginal_safe_dist[0]
