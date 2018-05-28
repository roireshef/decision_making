from logging import Logger

import numpy as np

from decision_making.src.global_constants import LON_ACC_LIMITS
from decision_making.src.planning.behavioral.action_space.action_space import ActionSpaceContainer
from decision_making.src.planning.behavioral.action_space.dynamic_action_space import DynamicActionSpace
from decision_making.src.planning.behavioral.action_space.static_action_space import StaticActionSpace
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState, RelativeLane
from decision_making.src.planning.behavioral.default_config import DEFAULT_STATIC_RECIPE_FILTERING, \
    DEFAULT_DYNAMIC_RECIPE_FILTERING
from decision_making.src.planning.behavioral.data_objects import AggressivenessLevel, StaticActionRecipe
from decision_making.src.planning.behavioral.evaluators.heuristic_action_spec_evaluator import \
    HeuristicActionSpecEvaluator
from decision_making.src.planning.behavioral.evaluators.naive_value_approximator import NaiveValueApproximator
from decision_making.src.planning.behavioral.evaluators.velocity_profile import VelocityProfile
from decision_making.src.planning.behavioral.filtering.action_spec_filter_bank import FilterIfNone
from decision_making.src.planning.behavioral.filtering.action_spec_filtering import ActionSpecFiltering
from decision_making.src.planning.behavioral.planner.single_step_behavioral_planner import SingleStepBehavioralPlanner
from decision_making.src.planning.utils.map_utils import MapUtils
from decision_making.src.planning.utils.math import Math
from decision_making.src.planning.utils.optimal_control.poly1d import QuarticPoly1D
from decision_making.src.prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.state.state import ObjectSize, EgoState, State
from mapping.src.service.map_service import MapService



# test Specify, when ego starts with velocity very close to the target velocity
def test_specifyGoal_closeToTargetVelocity_specifyNotFail():

    logger = Logger("test_specifyStaticAction")
    road_id = 20
    ego_lon = 400.
    lane_width = MapService.get_instance().get_road(road_id).lane_width
    road_mid_lat = MapService.get_instance().get_road(road_id).lanes_num * lane_width / 2
    size = ObjectSize(4, 2, 1)

    action_space = StaticActionSpace(logger, DEFAULT_STATIC_RECIPE_FILTERING)

    target_vel = action_space.recipes[0].velocity
    ego_vel = target_vel + 0.01
    ego_cpoint, ego_yaw = MapService.get_instance().convert_road_to_global_coordinates(road_id, ego_lon, road_mid_lat - lane_width)
    ego = EgoState(0, 0, ego_cpoint[0], ego_cpoint[1], ego_cpoint[2], ego_yaw, size, 0, ego_vel, 0, 0, 0, 0)

    state = State(None, [], ego)
    behavioral_state = BehavioralGridState.create_from_state(state, logger)

    action_specs = [action_space.specify_goal(recipe, behavioral_state) for i, recipe in enumerate(action_space.recipes)]

    specs = [action_specs[i] for i, recipe in enumerate(action_space.recipes)
                if recipe.relative_lane == RelativeLane.SAME_LANE and recipe.aggressiveness == AggressivenessLevel.CALM
                        and recipe.velocity == target_vel]

    # check specification of CALM SAME_LANE static action
    assert len(specs) > 0 and specs[0] is not None


# check if accelerations for different aggressiveness levels are in limits
# static action from 0 to 57 km/h
def test_specifyGoal_accelerationFrom0_atLeastOneActionInLimits():

    logger = Logger("test_specifyStaticAction")
    road_id = 20
    ego_lon = 400.
    lane_width = MapService.get_instance().get_road(road_id).lane_width
    road_mid_lat = MapService.get_instance().get_road(road_id).lanes_num * lane_width / 2
    size = ObjectSize(4, 2, 1)
    target_vel = 57/3.6

    action_space = StaticActionSpace(logger, DEFAULT_STATIC_RECIPE_FILTERING)
    action_space._recipes = np.array([StaticActionRecipe(RelativeLane.SAME_LANE, target_vel, AggressivenessLevel.CALM),
                                      StaticActionRecipe(RelativeLane.SAME_LANE, target_vel, AggressivenessLevel.STANDARD),
                                      StaticActionRecipe(RelativeLane.SAME_LANE, target_vel, AggressivenessLevel.AGGRESSIVE)])

    ego_vel = 0
    ego_cpoint, ego_yaw = MapService.get_instance().convert_road_to_global_coordinates(road_id, ego_lon, road_mid_lat - lane_width)
    ego = EgoState(0, 0, ego_cpoint[0], ego_cpoint[1], ego_cpoint[2], ego_yaw, size, 0, ego_vel, 0, 0, 0, 0)

    state = State(None, [], ego)
    behavioral_state = BehavioralGridState.create_from_state(state, logger)

    action_specs = [action_space.specify_goal(recipe, behavioral_state) for i, recipe in enumerate(action_space.recipes)]

    # at least one spec should be in limits
    in_limits = False
    for spec in action_specs:
        if spec is not None:
            A = QuarticPoly1D.time_constraints_matrix(spec.t)
            A_inv = np.linalg.inv(A)
            constraints = np.array([np.array([0, 0, 0, target_vel, 0])])
            poly_coefs = QuarticPoly1D.solve(A_inv, constraints)[0]
            in_limits = in_limits or QuarticPoly1D.is_acceleration_in_limits(poly_coefs, spec.t, LON_ACC_LIMITS)
    assert in_limits


# check if accelerations for different aggressiveness levels cover various accelerations
# test static actions from 9 to 14 m/s
def test_specifyGoal_acceleration_accelerationsCoverage():

    logger = Logger("test_specifyStaticAction")
    road_id = 20
    ego_lon = 400.
    lane_width = MapService.get_instance().get_road(road_id).lane_width
    road_mid_lat = MapService.get_instance().get_road(road_id).lanes_num * lane_width / 2
    size = ObjectSize(4, 2, 1)
    target_vel = 14

    action_space = StaticActionSpace(logger, DEFAULT_STATIC_RECIPE_FILTERING)
    action_space._recipes = np.array([StaticActionRecipe(RelativeLane.SAME_LANE, target_vel, AggressivenessLevel.CALM),
                                      StaticActionRecipe(RelativeLane.SAME_LANE, target_vel, AggressivenessLevel.STANDARD),
                                      StaticActionRecipe(RelativeLane.SAME_LANE, target_vel, AggressivenessLevel.AGGRESSIVE)])

    # verify the specification times are not too long for accelerations from 32 to 50 km/h
    ego_vel = 9
    ego_cpoint, ego_yaw = MapService.get_instance().convert_road_to_global_coordinates(road_id, ego_lon, road_mid_lat - lane_width)
    ego = EgoState(0, 0, ego_cpoint[0], ego_cpoint[1], ego_cpoint[2], ego_yaw, size, 0, ego_vel, 0, 0, 0, 0)

    state = State(None, [], ego)
    behavioral_state = BehavioralGridState.create_from_state(state, logger)

    action_specs = [action_space.specify_goal(recipe, behavioral_state) for i, recipe in enumerate(action_space.recipes)]

    max_acc = []
    for spec in action_specs:
        if spec is not None:
            A = QuarticPoly1D.time_constraints_matrix(spec.t)
            A_inv = np.linalg.inv(A)
            constraints = np.array([np.array([0, ego_vel, 0, target_vel, 0])])
            poly_coefs = QuarticPoly1D.solve(A_inv, constraints)
            acc_poly = Math.polyder2d(poly_coefs, m=2)
            T_vals = np.arange(0., spec.t + 0.001, 0.1)
            acc_values = Math.polyval2d(acc_poly, T_vals)[0]
            max_acc.append(np.max(acc_values))

    # calm acceleration       = [0.4 - 0.8]
    # standard acceleration   = [0.8 - 1.5]
    # aggressive acceleration = [1.5 - 3.0]
    assert len(max_acc) >= 3
    assert max_acc[2] / max_acc[1] < 3
    assert max_acc[1] / max_acc[0] < 3


def test_specifyGoal_findSafeNotAggressiveAction():
    """
    Ego velocity 23 m/s, object velocity 14 m/s, the distance 92 m.
    Verify specifications coverage and existence of at least one static action, that is not too aggressive and safe.
    """
    logger = Logger("test_specifyGoal_static")
    road_id = 20
    ego_lon = 400.
    lane_width = MapService.get_instance().get_road(road_id).lane_width
    length = 4
    size = ObjectSize(length, 2, 1)

    action_space = ActionSpaceContainer(logger, [StaticActionSpace(logger, DEFAULT_STATIC_RECIPE_FILTERING)])
    spec_evaluator = HeuristicActionSpecEvaluator(logger)
    action_spec_validator = ActionSpecFiltering([FilterIfNone()], logger)
    road_frenet = MapUtils.get_road_rhs_frenet_by_road_id(road_id)

    ego_vel = 23
    F_vel = 14
    dist = 92

    ego = MapUtils.create_canonic_ego(0, ego_lon, lane_width / 2, ego_vel, size, road_frenet)
    F_lon = ego_lon + dist
    F = MapUtils.create_canonic_object(1, 0, F_lon, lane_width / 2, F_vel, size, road_frenet)

    state = State(None, [F], ego)
    behavioral_state = BehavioralGridState.create_from_state(state, logger)
    recipes = action_space.recipes
    recipes_mask = action_space.filter_recipes(recipes, behavioral_state)

    for i, recipe in enumerate(recipes):
        if recipe.relative_lane != RelativeLane.SAME_LANE or not np.isclose(recipe.velocity, 50 / 3.6):
            recipes_mask[i] = False

    # Action specification
    specs = np.full(recipes.__len__(), None)
    valid_action_recipes = [recipe for i, recipe in enumerate(recipes) if recipes_mask[i]]
    specs[recipes_mask] = action_space.specify_goals(valid_action_recipes, behavioral_state)

    specs_mask = action_spec_validator.filter_action_specs(specs, behavioral_state)
    valid_idxs = np.where(specs_mask)[0]
    assert len(valid_idxs) > 1
    assert specs[valid_idxs][0].t / specs[valid_idxs][1].t < 3

    # State-Action Evaluation
    costs = spec_evaluator.evaluate(behavioral_state, recipes, list(specs), specs_mask)
    safe_not_aggressive_indices = []
    for i in valid_idxs:
        if specs[i].t > 5 and costs[i] < np.inf:
            safe_not_aggressive_indices.append(i)
    assert len(safe_not_aggressive_indices) > 0
