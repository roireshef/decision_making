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
from decision_making.src.planning.behavioral.state.lane_change_state import LaneChangeState
from decision_making.src.planning.behavioral.data_objects import StaticActionRecipe, DynamicActionRecipe, \
    ActionSpec, RelativeLane, RelativeLongitudinalPosition, ActionType
from decision_making.src.planning.behavioral.default_config import DEFAULT_STATIC_RECIPE_FILTERING, \
    DEFAULT_DYNAMIC_RECIPE_FILTERING, DEFAULT_ACTION_SPEC_FILTERING, DEFAULT_ROAD_SIGN_RECIPE_FILTERING
from decision_making.src.planning.behavioral.planner.base_planner import BasePlanner
from logging import Logger

from decision_making.src.planning.behavioral.state.lane_merge_state import LaneMergeState
from decision_making.src.planning.types import ActionSpecArray, FS_SX
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

    def _create_action_specs(self, behavioral_state: BehavioralGridState) -> ActionSpecArray:
        """
        see base class
        """
        action_recipes = self.action_space.recipes

        # Recipe filtering
        recipes_mask = self.action_space.filter_recipes(action_recipes, behavioral_state)
        self.logger.debug('Number of actions originally: %d, valid: %d',
                          self.action_space.action_space_size, np.sum(recipes_mask))

        action_specs = np.full(len(action_recipes), None)
        valid_action_recipes = [action_recipe for i, action_recipe in enumerate(action_recipes) if recipes_mask[i]]
        action_specs[recipes_mask] = self.action_space.specify_goals(valid_action_recipes, behavioral_state)

        # choose the preferred lane
        target_lane = SingleStepBehavioralPlanner.choose_target_lane(behavioral_state)
        if target_lane != RelativeLane.SAME_LANE:
            # if chosen_action is doing nothing, then evaluator will choose lane-change action from the default actions
            #ego_vel = behavioral_state.ego_state.velocity
            #filter_results = np.full(len(action_specs), False)
            #if chosen_action.t < 4 and abs(ego_vel - chosen_action.v_T) < RuleBasedLaneMergePlanner.VEL_GRID_RESOLUTION/2:
            #print('TRY TO CHANGE LANE')
            target_lane_mask = np.array([spec is not None and spec.relative_lane == target_lane for spec in action_specs])
            action_specs[~target_lane_mask] = None

            # check safety for all standard actions to target_lane
            action_spec_filter = ActionSpecFiltering(
                filters=[FilterIfNone(), FilterForSafetyTowardsTargetVehicle(self.logger)], logger=self.logger)
            filter_results = action_spec_filter.filter_action_specs(action_specs, behavioral_state)

            if filter_results.any():
                print('CHANGE LANE')
                filter_results = action_spec_filter.filter_action_specs(action_specs, behavioral_state)
            else:  # continue with search on SAME_LANE
                lane_merge_state = LaneMergeState.create_from_behavioral_state(behavioral_state, target_lane)
                planner = RuleBasedLaneMergePlanner(self.logger)
                actions = planner._create_action_specs(lane_merge_state)
                filtered_actions = planner._filter_actions(lane_merge_state, actions)
                costs = planner._evaluate_actions(lane_merge_state, None, filtered_actions)
                chosen_action = filtered_actions[np.argmin(costs)]
                print('target_lane = ', target_lane, 'Chosen action: idx', np.argmin(costs),
                      [spec.__str__() for spec in chosen_action.action_specs])

                follow_lane_idx = [i for i, recipe in enumerate(action_recipes)
                                   if recipe.action_type == ActionType.FOLLOW_LANE and recipe.relative_lane == RelativeLane.SAME_LANE][0]
                action_specs[:] = None
                action_specs[follow_lane_idx] = chosen_action.action_specs[0].to_spec(lane_merge_state.ego_fstate_1d[FS_SX])

        # TODO: FOR DEBUG PURPOSES!
        num_of_considered_static_actions = sum(isinstance(x, StaticActionRecipe) for x in valid_action_recipes)
        num_of_considered_dynamic_actions = sum(isinstance(x, DynamicActionRecipe) for x in valid_action_recipes)
        num_of_specified_actions = sum(x is not None for x in action_specs)
        self.logger.debug('Number of actions specified: %d (#%dS,#%dD)',
                          num_of_specified_actions, num_of_considered_static_actions, num_of_considered_dynamic_actions)
        return action_specs

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

    @staticmethod
    def choose_target_lane(behavioral_state: BehavioralGridState) -> RelativeLane:
        ego_lane = behavioral_state.ego_state.map_state.lane_id
        speed_limit = MapUtils.get_lane(ego_lane).e_v_nominal_speed

        right_front = (RelativeLane.RIGHT_LANE, RelativeLongitudinalPosition.FRONT)
        if RelativeLane.RIGHT_LANE in behavioral_state.extended_lane_frames and right_front not in behavioral_state.road_occupancy_grid:
            return RelativeLane.RIGHT_LANE
        if right_front in behavioral_state.road_occupancy_grid:
            right_actor = behavioral_state.road_occupancy_grid[right_front][0].dynamic_object
            if right_actor.velocity > speed_limit - 0.5:
                return RelativeLane.RIGHT_LANE

        same_front = (RelativeLane.SAME_LANE, RelativeLongitudinalPosition.FRONT)
        if same_front not in behavioral_state.road_occupancy_grid:
            print('NO FRONT ACTOR')
        else:
            print('FRONT ACTOR at ', behavioral_state.road_occupancy_grid[same_front][0].longitudinal_distance)

        if RelativeLane.LEFT_LANE not in behavioral_state.extended_lane_frames or same_front not in behavioral_state.road_occupancy_grid:
            return RelativeLane.SAME_LANE

        front_actor = behavioral_state.road_occupancy_grid[same_front][0].dynamic_object
        if front_actor.velocity > speed_limit - 2:
            return RelativeLane.SAME_LANE

        left_front = (RelativeLane.LEFT_LANE, RelativeLongitudinalPosition.FRONT)
        if left_front in behavioral_state.road_occupancy_grid:
            left_actor = behavioral_state.road_occupancy_grid[left_front][0].dynamic_object
            if left_actor.velocity - front_actor.velocity < 2:
                return RelativeLane.SAME_LANE

        return RelativeLane.LEFT_LANE
