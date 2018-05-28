from logging import Logger

import numpy as np

from decision_making.src.global_constants import LON_ACC_LIMITS
from decision_making.src.planning.behavioral.action_space.static_action_space import StaticActionSpace
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState, RelativeLane
from decision_making.src.planning.behavioral.default_config import DEFAULT_STATIC_RECIPE_FILTERING
from decision_making.src.planning.behavioral.data_objects import AggressivenessLevel, StaticActionRecipe
from decision_making.src.planning.utils.math import Math
from decision_making.src.planning.utils.optimal_control.poly1d import QuarticPoly1D
from decision_making.src.state.state import ObjectSize, EgoState, State
from mapping.src.service.map_service import MapService


# test Specify, when ego starts with velocity very close to the target velocity
def test_specifyGoals_closeToTargetVelocity_specifyNotFail():
    logger = Logger("test_specifyStaticAction")
    road_id = 20
    ego_lon = 400.
    lane_width = MapService.get_instance().get_road(road_id).lane_width
    road_mid_lat = MapService.get_instance().get_road(road_id).lanes_num * lane_width / 2
    size = ObjectSize(4, 2, 1)

    action_space = StaticActionSpace(logger, DEFAULT_STATIC_RECIPE_FILTERING)

    target_vel = action_space.recipes[0].velocity
    ego_vel = target_vel + 0.01
    ego_cpoint, ego_yaw = MapService.get_instance().convert_road_to_global_coordinates(road_id, ego_lon,
                                                                                       road_mid_lat - lane_width)
    ego = EgoState(0, 0, ego_cpoint[0], ego_cpoint[1], ego_cpoint[2], ego_yaw, size, 0, ego_vel, 0, 0, 0, 0)

    state = State(None, [], ego)
    behavioral_state = BehavioralGridState.create_from_state(state, logger)

    action_specs = action_space.specify_goals(action_space.recipes, behavioral_state)

    specs = [action_specs[i] for i, recipe in enumerate(action_space.recipes)
             if recipe.relative_lane == RelativeLane.SAME_LANE and recipe.aggressiveness == AggressivenessLevel.CALM
             and recipe.velocity == target_vel]

    # check specification of CALM SAME_LANE static action
    assert len(specs) > 0 and specs[0] is not None


# check if accelerations for different aggressiveness levels are in limits
# static action from 0 to 57 km/h
def test_specifyGoals_accelerationFrom0_atLeastOneActionInLimits():
    logger = Logger("test_specifyStaticAction")
    road_id = 20
    ego_lon = 400.
    lane_width = MapService.get_instance().get_road(road_id).lane_width
    road_mid_lat = MapService.get_instance().get_road(road_id).lanes_num * lane_width / 2
    size = ObjectSize(4, 2, 1)
    target_vel = 47 / 3.6

    action_space = StaticActionSpace(logger, DEFAULT_STATIC_RECIPE_FILTERING)
    action_space._recipes = np.array([StaticActionRecipe(RelativeLane.SAME_LANE, target_vel, AggressivenessLevel.CALM),
                                      StaticActionRecipe(RelativeLane.SAME_LANE, target_vel,
                                                         AggressivenessLevel.STANDARD),
                                      StaticActionRecipe(RelativeLane.SAME_LANE, target_vel,
                                                         AggressivenessLevel.AGGRESSIVE)])

    ego_vel = 0
    ego_cpoint, ego_yaw = MapService.get_instance().convert_road_to_global_coordinates(road_id, ego_lon,
                                                                                       road_mid_lat - lane_width)
    ego = EgoState(0, 0, ego_cpoint[0], ego_cpoint[1], ego_cpoint[2], ego_yaw, size, 0, ego_vel, 0, 0, 0, 0)

    state = State(None, [], ego)
    behavioral_state = BehavioralGridState.create_from_state(state, logger)
    action_recipes = action_space.recipes
    recipes_mask = action_space.filter_recipes(action_recipes, behavioral_state)
    action_specs = np.full(action_recipes.__len__(), None)
    valid_action_recipes = [action_recipe for i, action_recipe in enumerate(action_recipes) if recipes_mask[i]]
    action_specs[recipes_mask] = action_space.specify_goals(valid_action_recipes, behavioral_state)

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
def test_specifyGoals_acceleration_accelerationsCoverage():
    logger = Logger("test_specifyStaticAction")
    road_id = 20
    ego_lon = 400.
    lane_width = MapService.get_instance().get_road(road_id).lane_width
    road_mid_lat = MapService.get_instance().get_road(road_id).lanes_num * lane_width / 2
    size = ObjectSize(4, 2, 1)
    target_vel = 14

    action_space = StaticActionSpace(logger, DEFAULT_STATIC_RECIPE_FILTERING)
    action_space._recipes = np.array([StaticActionRecipe(RelativeLane.SAME_LANE, target_vel, AggressivenessLevel.CALM),
                                      StaticActionRecipe(RelativeLane.SAME_LANE, target_vel,
                                                         AggressivenessLevel.STANDARD),
                                      StaticActionRecipe(RelativeLane.SAME_LANE, target_vel,
                                                         AggressivenessLevel.AGGRESSIVE)])

    # verify the specification times are not too long for accelerations from 32 to 50 km/h
    ego_vel = 9
    ego_cpoint, ego_yaw = MapService.get_instance().convert_road_to_global_coordinates(road_id, ego_lon,
                                                                                       road_mid_lat - lane_width)
    ego = EgoState(0, 0, ego_cpoint[0], ego_cpoint[1], ego_cpoint[2], ego_yaw, size, 0, ego_vel, 0, 0, 0, 0)

    state = State(None, [], ego)
    behavioral_state = BehavioralGridState.create_from_state(state, logger)
    action_recipes = action_space.recipes
    recipes_mask = action_space.filter_recipes(action_recipes, behavioral_state)
    action_specs = np.full(action_recipes.__len__(), None)
    valid_action_recipes = [action_recipe for i, action_recipe in enumerate(action_recipes) if recipes_mask[i]]
    action_specs[recipes_mask] = action_space.specify_goals(valid_action_recipes, behavioral_state)

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
