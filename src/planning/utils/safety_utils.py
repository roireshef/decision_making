import numpy as np
from logging import Logger
from decision_making.src.global_constants import LON_ACC_LIMITS, EPS, TRAJECTORY_TIME_RESOLUTION
from decision_making.src.planning.types import FrenetTrajectories2D, FrenetTrajectory2D, FS_SX, FS_SV, FS_DX
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor

LON_SAFETY_ACCEL_DURING_RESPONSE = 0.
LON_SAFETY_BACK_ACTOR_MAX_DECEL = 3.8


class SafetyRSS:

    @staticmethod
    def get_lon_safe_dist(ego_trajectories: FrenetTrajectories2D, ego_response_time: float,
                          obj_trajectory: FrenetTrajectory2D, obj_response_time: float,
                          from_time_idx: np.array, till_time_idx: np.array, margin: float, front_actor: bool, logger: Logger,
                          ego_behind_max_brake: float = -LON_ACC_LIMITS[0],
                          ego_ahead_max_brake: float = LON_SAFETY_BACK_ACTOR_MAX_DECEL) -> np.array:
        """
        Calculate longitudinal safety between ego and another object for all timestamps.
        Longitudinal safety between two objects considers only their longitudinal data: longitude and longitudinal velocity.
        Longitudinal RSS formula considers distance reduction during the reaction time and difference between
        objects' braking distances.
        An object is defined safe if it's safe either longitudinally OR laterally.
        :param ego_trajectories: ego Frenet trajectories: 3D tensor of shape: traj_num x timestamps_num x 6
        :param ego_response_time: [sec] ego response time
        :param obj_trajectory: object's Frenet trajectory: 2D matrix: timestamps_num x 6
        :param obj_response_time: [sec] object's response time
        :param from_time_idx: array of first relevant time indices for each ego_trajectory
        :param till_time_idx: array of last relevant time indices for each ego_trajectory
        :param margin: [m] cars' lengths half sum
        :param front_actor: True if the actor is in front of ego
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
            ego_trajectories_s = ego_trajectories_s[np.newaxis]

        obj_lon, obj_vel, obj_lat = obj_trajectory[:, FS_SX], obj_trajectory[:, FS_SV], obj_trajectory[:, FS_DX]

        # set infinite s for irrelevant time indices
        irrelevant_s = -np.inf if front_actor else np.inf
        for ego_s, from_i, till_i in zip(ego_lon, from_time_idx, till_time_idx):
            ego_s[till_i:] = irrelevant_s
            ego_s[:from_i] = irrelevant_s

        if front_actor:
            # extrapolate ego trajectories ego_response_time seconds beyond their end state
            dt = TRAJECTORY_TIME_RESOLUTION
            predictor = RoadFollowingPredictor(logger)
            extrapolated_times = np.arange(dt, ego_response_time + EPS, dt)
            last_ego_states_s = ego_trajectories_s[range(ego_trajectories_s.shape[0]), till_time_idx - 1]
            ego_extrapolation = predictor.predict_1d_frenet_states(last_ego_states_s, extrapolated_times)
            delay_shift = ego_extrapolation.shape[1]

            ext_ego_lon = np.concatenate((ego_lon, np.zeros_like(ego_extrapolation[..., FS_SX])), axis=1)
            ext_ego_vel = np.concatenate((ego_vel, np.zeros_like(ego_extrapolation[..., FS_SV])), axis=1)
            for i in range(delay_shift):
                ext_ego_lon[range(ext_ego_lon.shape[0]), till_time_idx + i] = ego_extrapolation[:, i, FS_SX]
                ext_ego_vel[range(ext_ego_vel.shape[0]), till_time_idx + i] = ego_extrapolation[:, i, FS_SV]

            # we assume ego continues its trajectory during its reaction time, so we compute the difference between
            # object's braking distance from any moment and delayed braking distance of ego
            braking_distances_diff = np.maximum(0, ext_ego_vel[:, delay_shift:] ** 2 - obj_vel ** 2) / (2 * ego_behind_max_brake)
            marginal_safe_dist = obj_lon - ext_ego_lon[:, delay_shift:] - braking_distances_diff - margin

        else:
            # The worst-case velocity of the rear object (either ego or another object) may increase during its reaction
            # time, since it may accelerate before it starts to brake.
            obj_vel_after_reaction_time = obj_vel + obj_response_time * LON_SAFETY_ACCEL_DURING_RESPONSE

            # longitudinal RSS formula considers distance reduction during the reaction time and difference between
            # objects' braking distances
            obj_acceleration_dist = 0.5 * LON_SAFETY_ACCEL_DURING_RESPONSE * obj_response_time ** 2
            min_safe_dist = np.maximum((obj_vel_after_reaction_time ** 2 - ego_vel ** 2) / (2 * ego_ahead_max_brake), 0) + \
                            (obj_vel * obj_response_time + obj_acceleration_dist) + margin
            marginal_safe_dist = ego_lon - obj_lon - min_safe_dist

        return marginal_safe_dist if ego_trajectories.ndim > 2 else marginal_safe_dist[0]
