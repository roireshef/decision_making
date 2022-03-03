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
        :param timestamps: a 2d numpy array of timestamps (relative to current timestamp) to check safety in, where
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
        max_lane_speed = state.self_gff.get_speed_limit_for_lane_segment(state.ego_state.lane_segment)
        relevant_velocity_limit = 0 if lon_direction == LongitudinalDirection.AHEAD else max_lane_speed

        # add dummy actors in cases no actor is perceived in the horizon
        if lon_direction == LongitudinalDirection.BEHIND:
            for lane, actor in lane_to_actor.items():
                if actor is None:
                    lane_to_actor[lane] = EgoCentricActorState(
                        actor_id="safety_dummy", velocity=relevant_velocity_limit, acceleration=0,
                        size=VehicleSize(0, 0), s_relative_to_ego=LongitudinalDirection.BEHIND.value*perception_horizon,
                        lane_difference=lane.value)

        # condition array - True if an action has an actor to check safety against, False otherwise
        action_actors = NumpyUtils.from_list_of_tuples([lane_to_actor[target_lane] for target_lane in lanes])
        action_has_actor = np.not_equal(action_actors, None)
        relevant_timestamps = timestamps[action_has_actor] if timestamps.shape[0] > 1 else timestamps

        # initialize result to True by default and if no participating actors for any action, return it
        safe = np.full_like(lanes, True, dtype=np.bool)
        if not np.any(action_has_actor):
            return safe

        # For each relevant action (has actor), get the actor's station (over all timestamps) and velocity IN EGO FRAME.
        # NOTE: The prediction for actors assumes worst-case scenario (applying worst_case_acc!)
        worst_case_acc = -rss_params.decel_front if lon_direction == LongitudinalDirection.AHEAD else rss_params.accel_rear
        relevant_actors_poly_s = np.array([[worst_case_acc/2, actor.velocity, actor.s_relative_to_ego]
                                           for actor in action_actors[action_has_actor]])

        relevant_actors_fstates = Poly1D.zip_polyval_with_derivatives(relevant_actors_poly_s, relevant_timestamps)
        relevant_actors_s, relevant_actors_v, _ = relevant_actors_fstates.transpose((2, 0, 1))

        # This code block compensates for velocity profiles of actors that go out of [0, max_lane_speed] range.
        # It does that by computing the time to arrival for those limit speeds, and then subtract a*t**/2 from s, as
        # well as clip v, with t equal to the leftovers from the limit time to the requested timestamps
        relevant_actors_T_s = np.array([[(relevant_velocity_limit - actor.velocity) / worst_case_acc]
                                        for actor in action_actors[action_has_actor]])
        actors_s_limit_fix = np.maximum(0, relevant_timestamps - relevant_actors_T_s) ** 2 * worst_case_acc / 2
        relevant_actors_s -= actors_s_limit_fix
        relevant_actors_v = np.clip(relevant_actors_v, 0, max_lane_speed)

        # for each relevant action (has actor), get the ego's station (over all timestamps)
        # RELATIVE TO CURRENT EGO FRAME (in which the other actors polynomials are represented)
        relevant_ego_fstates_gff_frame = Utils.sample_lon_frenet(
            trajectories[action_has_actor], relevant_timestamps)

        # Since previous command samples from trajectories that might be represented on different GFF than the
        # current ego GFF, we shift the longitudinal station values to be on current ego GFF
        relevant_ego_frame_on_gff_offset = np.array(
            [[state.ego_fstate_on_adj_lanes[state.gff_relativity_to_ego(trajectory.gff_id)][FS_SX], 0, 0]
             for trajectory in trajectories[action_has_actor]]
        )[:, np.newaxis, :]
        relevant_ego_fstates = relevant_ego_fstates_gff_frame - relevant_ego_frame_on_gff_offset

        # assign velocities for rear and front vehicles (actor/ego)
        relevant_rear_v, relevant_front_v = (relevant_ego_fstates[:, :, FS_SV], relevant_actors_v) \
            if lon_direction == LongitudinalDirection.AHEAD else (
            relevant_actors_v, relevant_ego_fstates[:, :, FS_SV])

        # compute distance between rear and front vehicles
        relevant_actors_len = np.array([actor.length for actor in action_actors[action_has_actor]])
        relevant_bumpers_distance = lon_direction.value * (relevant_actors_s - relevant_ego_fstates[:, :, FS_SX]) - \
                                    (state.ego_state.length + relevant_actors_len[:, np.newaxis]) / 2

        # compare RSS safety between rear and front vehicles
        relevant_pointwise_rss = SafetyUtils.check_pair_rss_safety(
            vel_front=relevant_front_v, vel_rear=relevant_rear_v, distance=relevant_bumpers_distance,
            rss_params=rss_params)

        # override safety result for all actions with an actor
        safe[action_has_actor] = np.all(relevant_pointwise_rss, axis=1)

        return safe

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
