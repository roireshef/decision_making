from abc import abstractmethod
from typing import List, Dict
from typing import Optional

import numpy as np
from decision_making.src.global_constants import BP_JERK_S_JERK_D_TIME_WEIGHTS
from decision_making.src.planning.behavioral.data_objects import AggressivenessLevel
from decision_making.src.planning.types import FS_2D_LEN, FrenetState2D
from decision_making.src.planning.utils.math_utils import Math
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D
from decision_making.src.rl_agent.environments.action_space.common.data_objects import LaneChangeEvent, RLActionRecipe
from decision_making.src.rl_agent.environments.state_space.common.data_objects import EgoCentricState
from decision_making.src.rl_agent.environments.uc_rl_map import RelativeLane


class LateralMixin:
    """ Mixin for lateral action abstraction implementation (used by TrajectoryBasedActionSpaceAdapter) """
    _params: Dict

    @abstractmethod
    def _get_lateral_targets(self, state: EgoCentricState, action_recipes: List[RLActionRecipe]) -> \
            (np.ndarray, np.ndarray, List[RelativeLane], List[LaneChangeEvent]):
        pass

    # UTILITY FUNCTIONS #

    @staticmethod
    def _project_ego_on_lanes(state: EgoCentricState, relative_lanes: List[RelativeLane]) -> np.ndarray:
        """
        Projects the ego state on adjacent lane for every relative lane given in <relative_lanes> list if it exists as
        a valid target lane.
        :param state: the state of the environment
        :param relative_lanes: the list of N relative lanes to project ego state on
        :return: numpy 2D array of shape Nx6. Returns np.nan wherever the relative lane is invalid
        """
        lane_valid = np.isin(np.array(relative_lanes), state.valid_target_lanes)

        projected_ego_fstates = np.full((len(relative_lanes), FS_2D_LEN), np.nan)
        projected_ego_fstates[lane_valid] = np.array([
            state.ego_fstate_on_adj_lanes[relative_lane]
            for relative_lane, is_valid in zip(relative_lanes, lane_valid) if is_valid
        ])
        return projected_ego_fstates

    @staticmethod
    def _get_relative_lane_list(is_left: Optional[np.ndarray] = None, is_same: Optional[np.ndarray] = None,
                                is_right: Optional[np.ndarray] = None, mask: Optional[np.ndarray] = None) \
            -> List[RelativeLane]:
        """
        Maps binary event vectors to list of LaneChangeEvents
        :param is_left: binary vector for the LEFT_LANE event
        :param is_same: binary vector for the SAME_LANE event
        :param is_right: binary vector for the RIGHT_LANE event
        :param mask: optional mask for output (where 1 will be a RelativeLane and 0 will be None)
        :return: a list of RelativeLanes corresponding to the binary vectors in the input
        """

        def get_or(obj, default):
            return obj if obj is not None else default

        code_vec = get_or(is_left, 0) * RelativeLane.LEFT_LANE.value + \
                   get_or(is_same, 0) * RelativeLane.SAME_LANE.value + \
                   get_or(is_right, 0) * RelativeLane.RIGHT_LANE.value

        return [RelativeLane(c) if m else None for c, m in zip(code_vec, get_or(mask, np.ones_like(code_vec)))]

    @staticmethod
    def _get_lane_change_event_list(is_keep: Optional[np.ndarray] = None, is_negotiate: Optional[np.ndarray] = None,
                                    is_commit: Optional[np.ndarray] = None, is_abort: Optional[np.ndarray] = None,
                                    mask: Optional[np.ndarray] = None) -> List[LaneChangeEvent]:
        """
        Maps binary event vectors to list of LaneChangeEvents
        :param is_keep: binary vector for the UNCHANGE event
        :param is_negotiate: binary vector for the NEGOTIATE event
        :param is_commit: binary vector for the COMMIT event
        :param is_abort: binary vector for the ABORT event
        :param mask: optional mask for output (where 1 will be a LaneChangeEvent and 0 will be None)
        :return: a list of LaneChangeEvents corresponding to the binary vectors in the input
        """

        def get_or(obj, default):
            return obj if obj is not None else default

        code_vec = get_or(is_keep, 0) * LaneChangeEvent.UNCHANGED.value + \
                   get_or(is_negotiate, 0) * LaneChangeEvent.NEGOTIATE.value + \
                   get_or(is_commit, 0) * LaneChangeEvent.COMMIT.value + \
                   get_or(is_abort, 0) * LaneChangeEvent.ABORT.value

        return [LaneChangeEvent(c) if m else None
                for c, m in zip(code_vec, get_or(mask, np.ones_like(code_vec)))]


class OnlyKeepLaneMixin(LateralMixin):
    """ Mixin for actions that don't change lanes """

    def _get_lateral_targets(self, state: EgoCentricState, action_recipes: List[RLActionRecipe]) -> \
            (np.ndarray, np.ndarray, List[RelativeLane], List[LaneChangeEvent]):
        dx_T, T_d = np.full(len(action_recipes), 0), np.full(len(action_recipes), 0)
        relative_lane = [RelativeLane.SAME_LANE] * len(action_recipes)
        event = [LaneChangeEvent.UNCHANGED] * len(action_recipes)

        return dx_T, T_d, relative_lane, event


class CommitLaneChangeMixin(LateralMixin):
    """ Mixin for actions that change lane by committing (moves between lane centers) """

    def _get_lateral_targets(self, state: EgoCentricState, action_recipes: List[RLActionRecipe]) -> \
            (np.ndarray, np.ndarray, List[RelativeLane], List[LaneChangeEvent]):
        """
        Get terminal lateral position dx_T and longitudinal horizon T_d for all action recipes, with no negotiation -
        lane change actions are based on pre-committing only.
        :param state: the current state of environment
        :param action_recipes: the recipes to get the lateral targets for
        :return: Tuple (absolute longitude of terminal goal, time for terminal goal, lane adjacency index relative to
        ego current gff that the lateral offset is <-1, 0, 1>)
        """
        ego_state = state.ego_state
        lc_state = ego_state.lane_change_state
        recipe_lateral_dir = np.array([recipe.lateral_dir.value for recipe in action_recipes])

        # Free lane driving (targets should be the negotiation offsets on the same lane)
        if lc_state.is_inactive:
            relative_lane, event = zip(*[[RelativeLane(recipe.lateral_dir.value),
                                          LaneChangeEvent(2 * abs(recipe.lateral_dir.value))]
                                         for recipe in action_recipes])
            # TODO: this doesn't account for lane-centring T_d when ego is not well centered
            T_d = np.abs(recipe_lateral_dir) * self._params["FULL_LANE_CHANGE_DURATION"]

        # in a middle of a lane change (commit)
        else:
            committing_left = np.full(len(action_recipes),
                                      lc_state.target_relative_lane == RelativeLane.LEFT_LANE, dtype=np.bool)
            on_source = np.full(len(action_recipes), lc_state.source_gff == ego_state.gff_id, dtype=np.bool)

            relative_lane = self._get_relative_lane_list(is_left=committing_left & on_source,
                                                         is_right=~committing_left & on_source,
                                                         is_same=~on_source,
                                                         mask=recipe_lateral_dir == 0)

            T_d = np.empty_like(recipe_lateral_dir) * np.nan
            T_d[recipe_lateral_dir == 0] = self._params["FULL_LANE_CHANGE_DURATION"] - \
                                           lc_state.time_since_commit(state.timestamp_in_sec)

            event = self._get_lane_change_event_list(is_keep=recipe_lateral_dir == 0,
                                                     mask=recipe_lateral_dir == 0)

        dx_T = np.zeros(len(action_recipes))
        return dx_T, T_d, relative_lane, event


class NegotiateLaneChangeMixin(LateralMixin):
    """ Mixin for actions that change lane with negotiation (2-phase process for lane change) """

    def _get_lateral_targets(self, state: EgoCentricState, action_recipes: List[RLActionRecipe]) -> \
            (np.ndarray, np.ndarray, List[Optional[RelativeLane]], List[Optional[LaneChangeEvent]]):
        """
        Determine ("specify") attributes of lateral transitions of given action recipes (target position dx_T,
        time horizon T_d, target lane, induced lane change event). This is mostly based on pre-determined transition
        times configured in the scenario configuration.
        :param state: the current state of environment
        :param action_recipes: the recipes to get the lateral targets for
        :return: Tuple (absolute longitude of terminal goal, time for terminal goal, lane adjacency index relative to
        ego current gff that the lateral offset is <-1, 0, 1>). In cases where lane does not exist or there's an issue
        with creating a lateral motion, the first two are np.nan and the last two are None.
        """
        # Map the current state of lane changing to the relevant handler
        if state.ego_state.lane_change_state.is_inactive:
            dx_T, T_d, relative_lane, event = self._get_lateral_targets_while_inactive(state, action_recipes)
        elif state.ego_state.lane_change_state.is_negotiating:
            dx_T, T_d, relative_lane, event = self._get_lateral_targets_while_negotiating(state, action_recipes)
        elif state.ego_state.lane_change_state.is_aborting:
            dx_T, T_d, relative_lane, event = self._get_lateral_targets_while_aborting(state, action_recipes)
        elif state.ego_state.lane_change_state.is_committing:
            dx_T, T_d, relative_lane, event = self._get_lateral_targets_while_committing(state, action_recipes)
        else:
            raise ValueError("lane change state is is invalid = %s" % str(state.ego_state.lane_change_state))

        # if target lane does not exist, invalidate results for this action
        for i, rel_lane in enumerate(relative_lane):
            if not rel_lane in state.valid_target_lanes:
                dx_T[i] = T_d[i] = np.nan
                relative_lane[i] = event[i] = None

        return dx_T, T_d, relative_lane, event

    def _get_lateral_targets_while_inactive(self, state: EgoCentricState, action_recipes: List[RLActionRecipe]) -> \
            (np.ndarray, np.ndarray, List[Optional[RelativeLane]], List[Optional[LaneChangeEvent]]):
        """
        Ego vehicle is currently lane centering. Lateral target offsets are either the negotiation offsets on the same
        lane (for recipes with lateral_direction!=0) or 0 (lane centering, for recipes with lateral_direction==0).
        Transition times are pre-determined (for transitioning to negotiation offset), or using jerk-optimal time.
        """
        recipe_lateral_dir = np.array([recipe.lateral_dir.value for recipe in action_recipes])

        T_d = np.abs(recipe_lateral_dir) * self._params["LANE_CENTER_TO_NEGOTIATE_OFFSET_DURATION"]
        T_d[recipe_lateral_dir == 0] = self._specify_lateral_times_for_lane_centering(state.ego_state.fstate)

        dx_T = recipe_lateral_dir * self._params["NEGOTIATE_OFFSET_DX"]
        relative_lane = [RelativeLane.SAME_LANE] * len(action_recipes)
        event = self._get_lane_change_event_list(is_keep=dx_T == 0, is_negotiate=dx_T != 0)

        return dx_T, T_d, relative_lane, event

    def _get_lateral_targets_while_negotiating(self, state: EgoCentricState, action_recipes: List[RLActionRecipe]) -> \
            (np.ndarray, np.ndarray, List[Optional[RelativeLane]], List[Optional[LaneChangeEvent]]):
        """
        Ego vehicle is currently (1) at the negotiation offset or (2) transitioning to it:
            (1) uses lateral transition duration T_d=0 since ego is already on the target (laterally)
            (2) uses time left - predetermined lane center -> neg. offset, minus time already travelled
        """
        lc_state = state.ego_state.lane_change_state
        recipe_lateral_dir = np.array([recipe.lateral_dir.value for recipe in action_recipes])
        time_since_start = lc_state.time_since_start(state.timestamp_in_sec)

        # (1) Already at the negotiation offset
        if lc_state.time_since_negotiation_offset_arrival(state.timestamp_in_sec) is not None:
            time_left_to_neg_offset = 0
            time_left_to_abort = self._params["LANE_CENTER_TO_NEGOTIATE_OFFSET_DURATION"]
            time_left_to_commit = self._params["NEGOTIATE_OFFSET_TO_COMMIT_DURATION"]
        # (2) On the way to negotiation offset
        else:
            time_left_to_neg_offset = self._params["LANE_CENTER_TO_NEGOTIATE_OFFSET_DURATION"] - time_since_start
            time_left_to_abort = time_since_start + self._params["DIRECTION_CHANGE_DURATION"]
            time_left_to_commit = self._params["FULL_LANE_CHANGE_DURATION"] - time_since_start

        assert time_left_to_neg_offset >= 0, "time_left_to_neg_offset=%s" % time_left_to_neg_offset
        assert time_left_to_abort >= 0, "time_left_to_abort=%s" % time_left_to_abort
        assert time_left_to_commit >= 0, "time_left_to_commit=%s" % time_left_to_commit

        # create binary indicators for the 3 cases
        current_direction = np.sign(lc_state.target_offset)
        action_directions = current_direction * recipe_lateral_dir
        is_keep = action_directions == 0
        is_commit = action_directions == 1
        is_abort = action_directions == -1

        T_d = is_keep * time_left_to_neg_offset + is_abort * time_left_to_abort + is_commit * time_left_to_commit
        dx_T = is_keep * lc_state.target_offset  # abort and commit are both dx_T=0
        relative_lane = list(map(RelativeLane, is_commit * recipe_lateral_dir))
        event = self._get_lane_change_event_list(is_keep=is_keep, is_commit=is_commit, is_abort=is_abort)

        return dx_T, T_d, relative_lane, event

    def _get_lateral_targets_while_aborting(self, state: EgoCentricState, action_recipes: List[RLActionRecipe]) -> \
            (np.ndarray, np.ndarray, List[Optional[RelativeLane]], List[Optional[LaneChangeEvent]]):
        """
        Ego vehicle is currently transitioning back to lane centering ("aborting"), either:
            (1) after it arrived and departed the negotiation offset - T_d is calculated relative to time since departure
            (2) after being interrupted on the way to the negotiation offset. In this case the agent has driven for the
                duration of:

                    T_org = lc_state.time_since_start - time_since_abort

                in the original direction before aborting. Then, from the time of aborting, it needs to complete:

                    T_abort_total = T_direction_change + T_org

                duration for aborting (we add dedicated duration for changing directions of travel, and for symmetry it
                has to go all the way back to lane center). Hence, duration left for completing the abort maneuver is:

                    T_left_to_abort = T_abort_total - time_since_abort =

                            lc_state.time_since_start + T_direction_change - 2 * time_since_abort
        """
        recipe_lateral_dir = np.array([recipe.lateral_dir.value for recipe in action_recipes])

        time_since_start = state.ego_state.lane_change_state.time_since_start(state.timestamp_in_sec)
        time_since_abort = state.ego_state.lane_change_state.time_since_abort(state.timestamp_in_sec)
        time_since_negotiation_offset_deprature = \
            state.ego_state.lane_change_state.time_since_negotiation_offset_deprature(state.timestamp_in_sec)

        assert time_since_negotiation_offset_deprature is None or time_since_negotiation_offset_deprature > 0, \
            "time_since_negotiation_offset_deprature is %f while aborting" % time_since_negotiation_offset_deprature

        # Originally departed from the negotiation offset
        if time_since_negotiation_offset_deprature:
            time_left_to_abort = self._params["LANE_CENTER_TO_NEGOTIATE_OFFSET_DURATION"] - time_since_abort

        # Interrupted on the way to the negotiation offset (before arriving there)
        else:
            time_left_to_abort = time_since_start + self._params["DIRECTION_CHANGE_DURATION"] - 2 * time_since_abort

        # T_d and dx_T are NaN for any action other than keep direction (and commit to lane merge)
        T_d = np.empty_like(recipe_lateral_dir) * np.nan
        T_d[recipe_lateral_dir == 0] = time_left_to_abort
        dx_T = T_d * 0

        relative_lane = self._get_relative_lane_list(is_same=recipe_lateral_dir == 0,
                                                     mask=recipe_lateral_dir == 0)
        event = self._get_lane_change_event_list(is_keep=recipe_lateral_dir == 0,
                                                 mask=recipe_lateral_dir == 0)

        return dx_T, T_d, relative_lane, event

    def _get_lateral_targets_while_committing(self, state: EgoCentricState, action_recipes: List[RLActionRecipe]) -> \
            (np.ndarray, np.ndarray, List[Optional[RelativeLane]], List[Optional[LaneChangeEvent]]):
        """
        Ego vehicle is currently committing a full lane change, either:
            (1) after it arrived and departed the negotiation offset - T_d is calculated relative to time since departure
            (2) after being interrupted on the way to the negotiation offset. In this case the agent will target
            completion of full lane change (lane-center to lane-center) in FULL_LANE_CHANGE_DURATION sec.
        """
        lc_state = state.ego_state.lane_change_state
        recipe_lateral_dir = np.array([recipe.lateral_dir.value for recipe in action_recipes])
        time_since_start = lc_state.time_since_start(state.timestamp_in_sec)
        time_since_neg_departure = lc_state.time_since_negotiation_offset_deprature(state.timestamp_in_sec)

        assert time_since_neg_departure is None or time_since_neg_departure > 0, \
            "time_since_negotiation_offset_deprature is %s while committing" % time_since_neg_departure

        # Originally departed from the negotiation offset
        if time_since_neg_departure:
            time_left_to_commit = self._params["NEGOTIATE_OFFSET_TO_COMMIT_DURATION"] - time_since_neg_departure

        # Interrupted on the way to the negotiation offset (before arriving there)
        else:
            time_left_to_commit = self._params["FULL_LANE_CHANGE_DURATION"] - time_since_start

        # T_d and dx_T are NaN for any action other than keep direction (and commit to lane merge)
        T_d = np.empty_like(recipe_lateral_dir) * np.nan
        T_d[recipe_lateral_dir == 0] = time_left_to_commit

        dx_T = T_d * 0

        is_committing_left = np.full_like(T_d, lc_state.target_relative_lane == RelativeLane.LEFT_LANE, dtype=np.bool)
        is_committing_right = ~is_committing_left
        is_on_source_gff = np.full_like(T_d, lc_state.source_gff == state.ego_state.gff_id, dtype=np.bool)

        relative_lane = self._get_relative_lane_list(is_left=is_committing_left & is_on_source_gff,
                                                     is_right=is_committing_right & is_on_source_gff,
                                                     is_same=~is_on_source_gff,
                                                     mask=recipe_lateral_dir == 0)
        event = self._get_lane_change_event_list(is_keep=recipe_lateral_dir == 0,
                                                 mask=recipe_lateral_dir == 0)

        return dx_T, T_d, relative_lane, event

    def _specify_lateral_times_for_lane_centering(self, ego_fstate: FrenetState2D) -> float:
        """
        Takes action recipes and ego state projection (on the correct target lanes) and specifies the lateral
        polynomial and terminal time (horizon) of motion for each recipe, by solving boundary conditions for the
        Jerk-optimal case.
        :param ego_fstate: the ego fstates for each recipe projected onto its corresponding target lane frame
        :return: 1D numpy array (shape 6) of lateral polynomial coefficients, terminal time
        """
        w_T, w_J = BP_JERK_S_JERK_D_TIME_WEIGHTS[AggressivenessLevel.STANDARD.value, [2, 1]][:, np.newaxis]

        *_, dx_0, dv_0, da_0 = ego_fstate[np.newaxis, :].transpose()
        dx_T = dv_T = np.zeros_like(dx_0)

        # T_s <- find minimal non-complex local optima within the BP_ACTION_T_LIMITS bounds, otherwise <np.nan>
        cost_coefs_d = QuinticPoly1D.time_cost_function_derivative_coefs(a_0=da_0, v_0=dv_0, v_T=dv_T, dx=dx_T, T_m=0,
                                                                         w_T=w_T, w_J=w_J)
        roots_d = Math.find_real_roots_in_limits(cost_coefs_d, np.array([.0, self._params['ACTION_MAX_TIME_HORIZON']]))
        T_d = np.fmin.reduce(roots_d, axis=-1)
        T_d[QuinticPoly1D.is_tracking_mode(dv_0, dv_T, da_0, dx_0, 0)] = 0

        return T_d.item()
