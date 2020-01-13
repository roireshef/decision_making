import numpy as np
from decision_making.src.messages.route_plan_message import RoutePlan
from decision_making.src.planning.behavioral.action_space.action_space import ActionSpaceContainer
from decision_making.src.planning.behavioral.action_space.dynamic_action_space import DynamicActionSpace
from decision_making.src.planning.behavioral.action_space.road_sign_action_space import RoadSignActionSpace
from decision_making.src.planning.behavioral.action_space.static_action_space import StaticActionSpace
from decision_making.src.planning.behavioral.evaluators.augmented_lane_action_spec_evaluator import \
    AugmentedLaneActionSpecEvaluator
from decision_making.src.planning.behavioral.filtering.action_spec_filter_bank import \
    FilterForSafetyTowardsTargetVehicle, FilterIfNone
from decision_making.src.planning.behavioral.filtering.action_spec_filtering import ActionSpecFiltering
from decision_making.src.planning.behavioral.planner.rule_based_lane_merge_planner import RuleBasedLaneMergePlanner
from decision_making.src.planning.behavioral.state.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.state.lane_change_state import LaneChangeState, LaneChangeStatus
from decision_making.src.planning.behavioral.data_objects import StaticActionRecipe, DynamicActionRecipe, \
    ActionSpec, RelativeLane, RelativeLongitudinalPosition, ActionType
from decision_making.src.planning.behavioral.default_config import DEFAULT_STATIC_RECIPE_FILTERING, \
    DEFAULT_DYNAMIC_RECIPE_FILTERING, DEFAULT_ACTION_SPEC_FILTERING, DEFAULT_ROAD_SIGN_RECIPE_FILTERING
from decision_making.src.planning.behavioral.planner.base_planner import BasePlanner
from logging import Logger

from decision_making.src.planning.behavioral.state.lane_merge_state import LaneMergeState
from decision_making.src.planning.types import ActionSpecArray, FS_SX, FS_DX, FS_DV
from decision_making.src.planning.utils.numpy_utils import NumpyUtils
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.state.state import State
from decision_making.src.utils.map_utils import MapUtils


class SingleStepBehavioralPlanner(BasePlanner):
    """
    For each received current-state:
     1.A behavioral, semantic state is created and its value is approximated.
     2.The full action-space is enumerated (recipes are generated).
     3.Recipes are filtered according to some pre-defined rules.
     4.Recipes are specified so ActionSpecs are created.
     5.Action Specs are evaluated.
     6.Lowest-Cost ActionSpec is chosen and its parameters are sent to TrajectoryPlanner.
    """
    def __init__(self, state: State, logger: Logger):
        super().__init__(logger)
        self.predictor = RoadFollowingPredictor(logger)

        speed_limit = MapUtils.get_lane(state.ego_state.map_state.lane_id).e_v_nominal_speed

        self.action_space = ActionSpaceContainer(logger, [StaticActionSpace(logger, DEFAULT_STATIC_RECIPE_FILTERING, speed_limit),
                                                          DynamicActionSpace(logger, self.predictor, DEFAULT_DYNAMIC_RECIPE_FILTERING),
                                                          RoadSignActionSpace(logger, self.predictor, DEFAULT_ROAD_SIGN_RECIPE_FILTERING)])

    def _create_behavioral_state(self, state: State, route_plan: RoutePlan, lane_change_state: LaneChangeState) -> BehavioralGridState:
        """
        Create behavioral state and update DIM state machine using same-lane GFF
        :param state: current state
        :param route_plan: route plan
        :return:
        """
        behavioral_state = BehavioralGridState.create_from_state(state=state, route_plan=route_plan,
                                                                 lane_change_state=lane_change_state,
                                                                 logger=self.logger)
        behavioral_state.update_dim_state()
        return behavioral_state

    def _create_action_specs(self, behavioral_state: BehavioralGridState, route_plan: RoutePlan) -> ActionSpecArray:
        """
        see base class
        """
        print('time:', behavioral_state.ego_state.timestamp_in_sec,
              'fstate on same lane:', NumpyUtils.str_log(behavioral_state.projected_ego_fstates[RelativeLane.SAME_LANE]),
              'lane change status:', behavioral_state.lane_change_state.status)

        # don't enable to change lane 10 seconds after last lane-change start
        if behavioral_state.lane_change_state.status != LaneChangeStatus.PENDING or \
                (behavioral_state.lane_change_state.lane_change_start_time is not None and
                 behavioral_state.ego_state.timestamp_in_sec - behavioral_state.lane_change_state.lane_change_start_time < 10):
            return self._specify_actions(behavioral_state)

        # choose the preferred lane by route_plan
        target_lane = SingleStepBehavioralPlanner.choose_target_lane(behavioral_state, route_plan)

        if target_lane == RelativeLane.SAME_LANE:  # if we don't want to change lane
            return self._specify_actions(behavioral_state, RelativeLane.SAME_LANE)  # use UC action space

        # perform RB optimization on the same lane
        lane_merge_state = LaneMergeState.create_from_behavioral_state(behavioral_state, target_lane)
        planner = RuleBasedLaneMergePlanner(None, self.logger)
        actions = planner._create_action_specs(lane_merge_state, route_plan)
        filtered_actions = planner._filter_actions(lane_merge_state, actions)
        costs = planner._evaluate_actions(lane_merge_state, route_plan, filtered_actions)
        RB_chosen_action = filtered_actions[np.argmin(costs)] if len(costs) > 0 else None

        if RB_chosen_action is None:  # if we can not change lane
            behavioral_state.lane_change_state.margin_to_keep_from_targets = \
                RuleBasedLaneMergePlanner.calculate_margin_to_keep_from_front(lane_merge_state)
            print('target_lane = ', target_lane, 'No actions found by search. Margin =', behavioral_state.lane_change_state.margin_to_keep_from_targets)
            return self._specify_actions(behavioral_state, RelativeLane.SAME_LANE)  # use UC action space
        else:
            print('target_lane = ', target_lane, 'Chosen action: idx', np.argmin(costs), [spec.__str__() for spec in RB_chosen_action.action_specs])

        # try to start lane change by checking safety of UC actions toward target_lane
        target_action_specs = self._specify_actions(behavioral_state, target_lane)  # specs to target_lane
        # check safety for all UC actions to target_lane
        action_spec_filter = ActionSpecFiltering(filters=[FilterIfNone(), FilterForSafetyTowardsTargetVehicle(None)], logger=None)
        UC_safe_actions = action_spec_filter.filter_action_specs(list(target_action_specs), behavioral_state)

        if UC_safe_actions.any() and RB_chosen_action.t < 5:  # there is a safe non-delayed UC action
            self._start_lane_change(behavioral_state, target_lane)
            return self._specify_actions(behavioral_state)  # specify all actions

        # continue with RB search on SAME_LANE
        if UC_safe_actions.any():
            print('************** SAFE BUT DELAYED **************')
        follow_lane_idx = [idx for idx, recipe in enumerate(self.action_space.recipes)
                           if recipe.relative_lane == RelativeLane.SAME_LANE][0]
        action_specs = np.full(len(self.action_space.recipes), None)
        action_specs[follow_lane_idx] = RB_chosen_action.action_specs[0].to_spec(lane_merge_state.ego_fstate_1d[FS_SX])
        return action_specs

    def _specify_actions(self, behavioral_state: BehavioralGridState, relative_lane: RelativeLane = None) -> ActionSpecArray:
        action_recipes = self.action_space.recipes

        # Recipe filtering
        recipes_mask = self.action_space.filter_recipes(action_recipes, behavioral_state)
        if relative_lane is not None:
            recipes_mask &= np.array([recipe.relative_lane == relative_lane for recipe in action_recipes])

        self.logger.debug('Number of actions originally: %d, valid: %d',
                          self.action_space.action_space_size, np.sum(recipes_mask))

        action_specs = np.full(len(action_recipes), None)
        valid_action_recipes = [action_recipe for i, action_recipe in enumerate(action_recipes) if recipes_mask[i]]
        action_specs[recipes_mask] = self.action_space.specify_goals(valid_action_recipes, behavioral_state)

        # TODO: FOR DEBUG PURPOSES!
        num_of_considered_static_actions = sum(isinstance(x, StaticActionRecipe) for x in valid_action_recipes)
        num_of_considered_dynamic_actions = sum(isinstance(x, DynamicActionRecipe) for x in valid_action_recipes)
        num_of_specified_actions = sum(x is not None for x in action_specs)
        self.logger.debug('Number of actions specified: %d (#%dS,#%dD)',
                          num_of_specified_actions, num_of_considered_static_actions, num_of_considered_dynamic_actions)
        return action_specs

    def _start_lane_change(self, behavioral_state: BehavioralGridState, target_lane: RelativeLane) -> None:
        print('************************************************  CHANGE LANE  ************************************************')
        behavioral_state.lane_change_state.lane_change_start_time = behavioral_state.ego_state.timestamp_in_sec
        behavioral_state.lane_change_state.status = LaneChangeStatus.ACTIVE_IN_SOURCE_LANE
        behavioral_state.lane_change_state.target_relative_lane = target_lane
        behavioral_state.lane_change_state.source_lane_gff = behavioral_state.extended_lane_frames[RelativeLane.SAME_LANE]
        behavioral_state.lane_change_state._target_lane_ids = behavioral_state.extended_lane_frames[target_lane].segment_ids
        behavioral_state.lane_change_state.autonomous_mode = True

    def _filter_actions(self, behavioral_state: BehavioralGridState, action_specs: ActionSpecArray) -> ActionSpecArray:
        """
        see base class
        """
        action_specs_mask = DEFAULT_ACTION_SPEC_FILTERING.filter_action_specs(action_specs, behavioral_state)
        filtered_action_specs = np.full(len(action_specs), None)
        filtered_action_specs[action_specs_mask] = action_specs[action_specs_mask]
        return filtered_action_specs

    def _evaluate_actions(self, behavioral_state: BehavioralGridState, route_plan: RoutePlan,
                          action_specs: ActionSpecArray) -> np.ndarray:
        """
        Evaluates Action-Specifications based on the following logic:
        * Only takes into account actions on RelativeLane.SAME_LANE
        * If there's a leading vehicle, try following it (ActionType.FOLLOW_VEHICLE, lowest aggressiveness possible)
        * If no action from the previous bullet is found valid, find the ActionType.FOLLOW_ROAD_SIGN action with lowest
        * aggressiveness, and save it.
        * Find the ActionType.FOLLOW_LANE action with maximal allowed velocity and lowest aggressiveness possible,
        * and save it.
        * Compare the saved FOLLOW_ROAD_SIGN and FOLLOW_LANE actions, and choose between them.
        :param action_specs: specifications of action_recipes.
        :return: numpy array of costs of semantic actions. Only one action gets a cost of 0, the rest get 1.
        """
        action_spec_evaluator = AugmentedLaneActionSpecEvaluator(self.logger)
        action_specs_exist = action_specs.astype(bool)
        return action_spec_evaluator.evaluate(behavioral_state, self.action_space.recipes, action_specs,
                                              list(action_specs_exist), route_plan)

    def _choose_action(self, behavioral_state: BehavioralGridState, action_specs: ActionSpecArray, costs: np.array) -> \
            ActionSpec:
        """
        see base class
        """
        return action_specs[np.argmin(costs)]

    #     # TODO: remove it
    #     action_idx = np.argmin(costs)
    #     chosen_spec = action_specs[action_idx]
    #     t = behavioral_state.ego_state.timestamp_in_sec
    #     print('chosen spec:', chosen_spec)
    #
    #     if t > 5:
    #         if chosen_spec.relative_lane == RelativeLane.LEFT_LANE and behavioral_state.lane_change_state.last_spec_idx is None:
    #             behavioral_state.lane_change_state.last_spec_idx = action_idx
    #             behavioral_state.lane_change_state.last_spec = chosen_spec
    #             print('save first: t, t_d =', t+chosen_spec.t, t+chosen_spec.t_d)
    #             self.save_trajectory(behavioral_state, chosen_spec, 'first.p')
    #
    #         last_idx = behavioral_state.lane_change_state.last_spec_idx
    #         if last_idx is not None and last_idx != action_idx and last_idx >= 0:
    #             last_spec = self.action_space.specify_goals([self.action_space.recipes[last_idx]], behavioral_state)[0]
    #             print('save second: t, t_d =', t+last_spec.t, t+last_spec.t_d)
    #             behavioral_state.lane_change_state.last_spec_idx = -1
    #             self.save_trajectory(behavioral_state, last_spec, 'second.p')
    #
    #     return chosen_spec
    #
    # # TODO: remove it
    # def save_trajectory(self, behavioral_state, spec, file):
    #     import pickle
    #     path = '/home/mz8cj6/temp/'
    #     t = behavioral_state.ego_state.timestamp_in_sec
    #     filtering = ActionSpecFiltering(filters=[], logger=self.logger)
    #     ftrajectories, ctrajectories = filtering._build_trajectories([spec], behavioral_state)
    #     ftrajectory, ctrajectory = ftrajectories[0], ctrajectories[0]
    #     till_idx = np.argmax(ftrajectory[:, FS_SX] == 0)
    #
    #     res = 0.1
    #     first_sample_time = (-t) % res
    #     EPS = np.finfo(np.float32).eps
    #     if res - first_sample_time < EPS:
    #         first_sample_time = 0
    #     time_samples = np.arange(first_sample_time, till_idx * res - np.finfo(np.float32).eps, res)
    #
    #     txy = np.c_[t + time_samples, ctrajectory[:till_idx, 0], ctrajectory[:till_idx, 1]]
    #     pickle.dump(txy, open(path + file, 'wb'))

    @staticmethod
    def choose_target_lane(behavioral_state: BehavioralGridState, route_plan: RoutePlan) -> RelativeLane:

        # if lane_fstate[FS_DX] * np.sign(lane_fstate[FS_DV]) < -1:  # toward lane center, i.e. after crossing the border
        #     behavioral_state.lane_change_state.lane_change_start_time = behavioral_state.ego_state.timestamp_in_sec
        # wait 8 sec after previous lane change
        ego_lane = behavioral_state.ego_state.map_state.lane_id
        speed_limit = MapUtils.get_lane(ego_lane).e_v_nominal_speed

        right_front = (RelativeLane.RIGHT_LANE, RelativeLongitudinalPosition.FRONT)
        if RelativeLane.RIGHT_LANE in behavioral_state.extended_lane_frames and right_front not in behavioral_state.road_occupancy_grid:
            print('RIGHT_LANE: no front car')
            return RelativeLane.RIGHT_LANE

        if right_front in behavioral_state.road_occupancy_grid:
            right_actor = behavioral_state.road_occupancy_grid[right_front][0].dynamic_object
            if right_actor.velocity > speed_limit - 0.5:
                print('RIGHT_LANE: right is fast:', right_actor.velocity)
                return RelativeLane.RIGHT_LANE

        same_front = (RelativeLane.SAME_LANE, RelativeLongitudinalPosition.FRONT)
        if same_front not in behavioral_state.road_occupancy_grid:
            print('NO FRONT ACTOR')
        else:
            print('FRONT ACTOR at ', behavioral_state.road_occupancy_grid[same_front][0].longitudinal_distance)

        if RelativeLane.LEFT_LANE not in behavioral_state.extended_lane_frames or same_front not in behavioral_state.road_occupancy_grid:
            print('SAME_LANE: same_front does not exist or left_lane does not exist')
            return RelativeLane.SAME_LANE

        front_actor = behavioral_state.road_occupancy_grid[same_front][0].dynamic_object
        if front_actor.velocity > speed_limit - 2:
            print('SAME_LANE: front actor is fast: ', front_actor.velocity)
            return RelativeLane.SAME_LANE

        left_front = (RelativeLane.LEFT_LANE, RelativeLongitudinalPosition.FRONT)
        if left_front in behavioral_state.road_occupancy_grid:
            left_actor = behavioral_state.road_occupancy_grid[left_front][0].dynamic_object
            if left_actor.velocity - front_actor.velocity < 2:
                print('SAME_LANE: left actor is not faster than front')
                return RelativeLane.SAME_LANE

        print('LEFT_LANE')
        return RelativeLane.LEFT_LANE
