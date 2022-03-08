from typing import Dict, Set
from typing import Union

import math
import numpy as np
from decision_making.src.planning.types import FS_SV, FS_SX
from decision_making.src.planning.utils.optimal_control.poly1d import Poly1D
from decision_making.src.rl_agent.environments.state_space.common.data_objects import EgoCentricActorState, EgoCentricState, \
    VehicleSize
from decision_making.src.rl_agent.environments.uc_rl_map import RelativeLane
from decision_making.src.rl_agent.global_types import LongitudinalDirection
from decision_making.src.rl_agent.global_types import RSSParams
from decision_making.src.rl_agent.utils.numpy_utils import NumpyUtils
from decision_making.src.rl_agent.utils.samplable_werling_trajectory import Utils


class SafetyUtils:
    @staticmethod
    def check_pair_rss_safety(vel_front: Union[float, np.ndarray], vel_rear: Union[float, np.ndarray],
                              distance: Union[float, np.ndarray], rss_params: RSSParams):
        """
        Verifies rss safety between two vehicles, one ahead of the other
        For the RSS condition -  see https://arxiv.org/pdf/1708.06374.pdf
        :param vel_front: current velocity of the front vehicle
        :param vel_rear: current velocity of the rear vehicle
        :param distance: current distance between the two vehicles
        :param rss_params: the rss params by which safety will be verified
        :return: True if rss safety conditions are met, else False
        """
        ro = rss_params.ro
        accel_rear = rss_params.accel_rear
        decel_rear = rss_params.decel_rear
        decel_front = rss_params.decel_front
        rear_progress = vel_rear * ro + (accel_rear * ro**2) / 2 + (vel_rear + ro * accel_rear)**2 / (2 * decel_rear)
        front_progress = vel_front**2 / (2 * decel_front)

        return np.maximum(rear_progress - front_progress, 0) <= distance

    @staticmethod
    def is_future_safe_directional(lon_direction: LongitudinalDirection, state: EgoCentricState, timestamps: np.ndarray,
                                   lanes: np.ndarray, trajectories: np.ndarray, rss_params: RSSParams,
                                   perception_horizon: float) -> np.ndarray:
        """
        For each action (represented by a target lane and a trajectory), this finds the relevant vehicle in <state> to
        check safety against in a single longitudinal direction given by <direction>. It returns True for each action
        if safety is maintained during <timestamps>, and False otherwise.
        Note that <target_lanes> and <trajectories> should are coupled and has the same length/order.
        Note that if timestamps is an empty array, then result is an array of True values.
        :param lon_direction: either check against an actor ahead or behind the host vehicle
        :param state: the state to pull actors information from
        :param timestamps: a 2d numpy array of timestamps (relative to state.timestamp_in_sec) to check safety in, where
        axis 0 corresponds to zip with trajectories and axis 1 is different times to check those trajectories. Axis 0
        can be just a wrapper - in that case the 1d array in axis 1 is replicated for all trajectories (all trajectories
        will be checked with the same timestamps)
        :param lanes: a 1d numpy array of dtype RelativeLane corresponding to trajectories of the actions under test
        :param trajectories: a 1d numpy array of dtype SamplableWerlingTrajectory corresponding to trajectories of the
        actions under test
        :param rss_params: the rss params by which safety will be verified
        :param perception_horizon: [m] if no actor is spotted on a lane, we place a dummy actor at the perc. horizon
        :return: a 1d numpy array of boolean type - True if an action passed safety check, False otherwise
        """
        if len(timestamps) == 0:
            return np.full(len(trajectories), True)

        assert len(timestamps.shape) == 2, "timestamps must be 2d"
        assert len(timestamps) == 1 or len(timestamps) == len(trajectories), \
            "timestamps and trajectories sizes don't match"

        unique_lanes = set(lanes)

        # create an array of critical actor for each action-under-test (given by: target lane, trajectory; directional)
        lane_to_actor = SafetyUtils.map_lane_to_closest_actor(state, unique_lanes, lon_direction)

        # compute relevant velocity limit that actors can reach to (either 0 or max_lane_speed) for which we compensate
        # their motion profiles below
        # TODO: change speed per actor
        lane_speed_limit = state.self_gff.get_speed_limit_for_lane_segment(state.ego_state.lane_segment)

        # add dummy actors in cases no actor is perceived in the horizon
        if lon_direction == LongitudinalDirection.BEHIND:
            for lane, actor in lane_to_actor.items():
                if actor is None:
                    lane_to_actor[lane] = EgoCentricActorState(
                        actor_id="safety_dummy",
                        velocity=0 if lon_direction == LongitudinalDirection.AHEAD else lane_speed_limit,
                        acceleration=0,
                        size=VehicleSize(0, 0),
                        s_relative_to_ego=LongitudinalDirection.BEHIND.value*perception_horizon,
                        lane_difference=lane.value)

        # condition array - True if an action has an actor to check safety against, False otherwise
        action_actors = NumpyUtils.from_list_of_tuples([lane_to_actor[target_lane] for target_lane in lanes])
        action_has_actor = np.not_equal(action_actors, None)

        # initialize result to True by default and if no participating actors for any action, return it
        safe = np.full_like(lanes, True, dtype=np.bool)
        if not np.any(action_has_actor):
            return safe

        # override safety result for all actions with an actor
        safe[action_has_actor] = SafetyUtils._is_future_safe_towards_actor(
            lon_direction=lon_direction,
            state=state,
            timestamps=timestamps[action_has_actor] if timestamps.shape[0] > 1 else timestamps,
            actors=action_actors[action_has_actor],
            trajectories=trajectories[action_has_actor],
            lane_speed_limit=lane_speed_limit,
            rss_params=rss_params
        )

        return safe

    @staticmethod
    def _is_future_safe_towards_actor(actors: np.ndarray, trajectories: np.ndarray, timestamps: np.ndarray,
                                      lon_direction: LongitudinalDirection, state: EgoCentricState,
                                      lane_speed_limit: float, rss_params: RSSParams) -> np.ndarray:
        """
        Given coupled <actors> and ego <trajectories>, for each coupled entry this method predicts both ego and actor
        in all <timestamps> and returns True if safety maintained throughout all timestamps, False otherwise. The
        prediction model for actor is worst-case model (with velocity clipped in the range [0, <lane_speed_limit>]),
        for the ego vehicle the model assumes perfect tracking of inteded trajectory (given by the corresponding entry
        from <trajectories>).

        :param actors: either check against an actor ahead or behind the host vehicle
        :param trajectories: a 1d numpy array of dtype SamplableWerlingTrajectory corresponding to trajectories of the
        actions under test
        :param timestamps: a 2d numpy array of timestamps relative to state.timestamp_in_sec to check safety in, where
        axis 0 corresponds to zip with trajectories and axis 1 is different times to check those trajectories. Axis 0
        can be just a wrapper - in that case the 1d array in axis 1 is replicated for all trajectories (all trajectories
        will be checked with the same timestamps)
        :param lon_direction: either check against an actor ahead or behind the host vehicle
        :param state: the state to pull actors information from
        :param lane_speed_limit: [m/s] the maximal assumed speed for actors' prediction model
        :param rss_params: the rss params by which safety will be verified
        """
        # For each relevant action (has actor), get the actor's station (over all timestamps) and velocity IN EGO FRAME.
        # NOTE: The prediction for actors assumes worst-case scenario (applying worst_case_acc!)
        worst_case_acc = -rss_params.decel_front if lon_direction == LongitudinalDirection.AHEAD else rss_params.accel_rear
        actors_poly_s = np.array([[worst_case_acc/2, actor.velocity, actor.s_relative_to_ego] for actor in actors])

        actors_fstates = Poly1D.zip_polyval_with_derivatives(actors_poly_s, timestamps)
        actors_s, actors_v, _ = actors_fstates.transpose((2, 0, 1))

        # This code block compensates for velocity profiles of actors that go out of [0, max_lane_speed] range.
        # It does that by computing the time to arrival for those limit speeds, and then subtract a*t**/2 from s, as
        # well as clip v, with t equal to the leftovers from the limit time to the requested timestamps
        speed_boundary = 0 if lon_direction == LongitudinalDirection.AHEAD else lane_speed_limit
        actors_T_s = np.array([[(speed_boundary - actor.velocity) / worst_case_acc] for actor in actors])
        actors_s_limit_fix = np.maximum(0, timestamps - actors_T_s) ** 2 * worst_case_acc / 2
        actors_s -= actors_s_limit_fix
        actors_v = np.clip(actors_v, 0, speed_boundary)

        # timestamps are originally specified relative to state.timestamp_in_sec. Since in what comes next we sample the
        # trajectories in relative time, we have to align the timestamps to the trajectories timestamp (might not be
        # the same). When calling this methods for filters, state and trajectories should already be aligned!
        assert len(set(trajectory.timestamp_in_sec for trajectory in trajectories)) == 1, \
            "current SafetyUtils._is_future_safe_to_actor implementation only works when all trajectories are aligned" \
            "to the same trajectory.timestamp_in_sec"
        trajectory_alinged_timestamps = timestamps + state.timestamp_in_sec - trajectories[0].timestamp_in_sec

        # for each entry, we represent the ego's 1D longitudinal fstate (over all timestamps) RELATIVE TO CURRENT
        # EGO FRAME (current localization), since actors polynomials are extracted relative to the current ego
        # localization. Since ego fstates sampled from trajectories that might be
        # represented on different GFF than the current ego GFF, we offset the longitudinal station values
        ego_fstates_on_trajectory_gff = Utils.sample_lon_frenet(trajectories, trajectory_alinged_timestamps)
        ego_fstates_on_current_ego_frame = ego_fstates_on_trajectory_gff - np.array([
            [state.ego_fstate_on_gffs[trajectory.gff_id][FS_SX], 0, 0]
            for trajectory in trajectories
        ])[:, np.newaxis, :]

        # assign velocities for rear and front vehicles (actor/ego)
        rear_v, front_v = (ego_fstates_on_current_ego_frame[:, :, FS_SV], actors_v) \
            if lon_direction == LongitudinalDirection.AHEAD \
            else (actors_v, ego_fstates_on_current_ego_frame[:, :, FS_SV])

        # compute distance between rear and front vehicles
        actors_len = np.array([actor.length for actor in actors])
        bumpers_distance = lon_direction.value * (actors_s - ego_fstates_on_current_ego_frame[:, :, FS_SX]) - \
                                    (state.ego_state.length + actors_len[:, np.newaxis]) / 2

        # compare RSS safety between rear and front vehicles for all timestamps
        pointwise_rss = SafetyUtils.check_pair_rss_safety(
            vel_front=front_v, vel_rear=rear_v, distance=bumpers_distance,
            rss_params=rss_params)

        # an entry is safe if safety is maintained throughout all timestamps
        return np.all(pointwise_rss, axis=1)

    @staticmethod
    def constant_acc_kinematics(x0: float, v0: float, a: float, t: float):
        """ given initial position and velocity at t=0, constant acceleration and T, this method returns the
        velocity and position at time t=T assuming constant acceleration throughout """
        return a, v0 + a*t, x0 + v0*t + a*t**2/2

    @staticmethod
    def rss_safety_directional(state: EgoCentricState, lane: RelativeLane, lon_direction: LongitudinalDirection,
                               rss_params: RSSParams) -> bool:
        """
        full-RSS check (both ego and actor are performing according to worst-case scenario).
        This method makes use of the fact state.behavioral_grid holds min-heaps (key is the absolute distance to ego)
        for efficiency
        """
        closest_actor = next(iter(state.behavioral_grid[(lane, lon_direction)]), None)

        if closest_actor is None:
            return True

        cg_to_bumper_margin = (state.ego_state.length + closest_actor.length) / 2
        bumper_to_bumper_dist_ahead = math.fabs(closest_actor.s_relative_to_ego) - cg_to_bumper_margin

        # assign velocities for rear and front vehicles (actor/ego)
        rear_v, front_v = (state.ego_state.fstate[FS_SV], closest_actor.velocity) \
            if lon_direction == LongitudinalDirection.AHEAD else (closest_actor.velocity, state.ego_state.fstate[FS_SV])

        return SafetyUtils.check_pair_rss_safety(vel_front=front_v, vel_rear=rear_v, rss_params=rss_params,
                                                 distance=bumper_to_bumper_dist_ahead)

    @staticmethod
    def map_lane_to_closest_actor(state: EgoCentricState, target_lanes: Set[RelativeLane],
                                  direction: LongitudinalDirection) -> Dict[RelativeLane, EgoCentricActorState]:
        """
        Returns a dictionary mapping target lanes at a longitudinal direction and their closest actors to ego vehicle.
        This method makes use of the fact state.behavioral_grid holds min-heaps (key is the absolute distance to ego)
        for efficiency.
        :param state: the state to pull actors information from
        :param target_lanes: set of unique RelativeLanes to consider
        :param direction: the longitudinal direction to consider actors towards
        :return: dictionary mapping a target lane to the closest actor (None if actor does not exist)
        """
        return {target_lane: next(iter(state.behavioral_grid[(target_lane, direction)]), None)
                for target_lane in target_lanes}
