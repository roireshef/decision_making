from logging import Logger
import numpy as np
import pytest

from decision_making.src.global_constants import LON_ACC_LIMITS, BP_ACTION_T_LIMITS
from decision_making.src.planning.behavioral.action_space.dynamic_action_space import DynamicActionSpace
from decision_making.src.planning.behavioral.action_space.static_action_space import StaticActionSpace
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState, RelativeLane, \
    RelativeLongitudinalPosition
from decision_making.src.planning.behavioral.constants import MIN_VELOCITY
from decision_making.src.planning.behavioral.data_objects import AggressivenessLevel, StaticActionRecipe
from decision_making.src.planning.trajectory.frenet_constraints import FrenetConstraints
from decision_making.src.planning.utils.numpy_utils import NumpyUtils
from decision_making.src.planning.utils.optimal_control.poly1d import QuarticPoly1D, Poly1D
from decision_making.src.prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.state.state import ObjectSize, EgoState, State
from mapping.src.service.map_service import MapService
from decision_making.src.planning.utils.math import Math


# test Specify from very close to target velocity
def test_staticActionSpace_specifyGoal_closeToTargetVelocityShouldNotFail():

    logger = Logger("test_specifyStaticAction")
    road_id = 20
    ego_lon = 400.
    lane_width = MapService.get_instance().get_road(road_id).lane_width
    road_mid_lat = MapService.get_instance().get_road(road_id).lanes_num * lane_width / 2
    size = ObjectSize(4, 2, 1)

    action_space = StaticActionSpace(logger)

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
def test_staticActionSpace_specifyGoal_accelerationsInLimits():

    logger = Logger("test_specifyStaticAction")
    road_id = 20
    ego_lon = 400.
    lane_width = MapService.get_instance().get_road(road_id).lane_width
    road_mid_lat = MapService.get_instance().get_road(road_id).lanes_num * lane_width / 2
    size = ObjectSize(4, 2, 1)
    target_vel = 57/3.6

    action_space = StaticActionSpace(logger)
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


# check if accelerations for different aggressiveness levels are reasonable and cover various accelerations
# static action from 9 to 14 m/s
def test_staticActionSpace_specifyGoal_accelerationsCoverage():

    logger = Logger("test_specifyStaticAction")
    road_id = 20
    ego_lon = 400.
    lane_width = MapService.get_instance().get_road(road_id).lane_width
    road_mid_lat = MapService.get_instance().get_road(road_id).lanes_num * lane_width / 2
    size = ObjectSize(4, 2, 1)
    target_vel = 14

    action_space = StaticActionSpace(logger)
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

    # very_calm_acceleration = 0.5   # 0-100 km/h in 56 sec
    # standard_acceleration = 1.0    # 0-100 km/h in 28 sec
    # aggressive_acceleration = 2.0  # 0-100 km/h in 14 sec
    assert max_acc[2] / max_acc[1] < 3
    assert max_acc[1] / max_acc[0] < 3


# test specify for dynamic action from a slightly unsafe position
def test_dynamicActionSpace_specifyGoal_unsafePosition():
    logger = Logger("test_specifyDynamicAction")
    road_id = 20
    ego_lon = 400.
    lane_width = MapService.get_instance().get_road(road_id).lane_width
    road_mid_lat = MapService.get_instance().get_road(road_id).lanes_num * lane_width / 2
    size = ObjectSize(4, 2, 1)

    predictor = RoadFollowingPredictor(logger)
    action_space = DynamicActionSpace(logger, predictor)

    # verify the peak acceleration does not exceed the limit by calculating the average acceleration from 0 to 50 km/h
    ego_vel = 10
    ego_cpoint, ego_yaw = MapService.get_instance().convert_road_to_global_coordinates(road_id, ego_lon, road_mid_lat - lane_width)
    ego = EgoState(0, 0, ego_cpoint[0], ego_cpoint[1], ego_cpoint[2], ego_yaw, size, 0, ego_vel, 0, 0, 0, 0)

    obj_vel = 10
    obj_lon = ego_lon + 20
    obj_cpoint, obj_yaw = MapService.get_instance().convert_road_to_global_coordinates(road_id, obj_lon, road_mid_lat - lane_width)
    obj = EgoState(0, 0, obj_cpoint[0], obj_cpoint[1], obj_cpoint[2], obj_yaw, size, 0, obj_vel, 0, 0, 0, 0)

    state = State(None, [obj], ego)
    behavioral_state = BehavioralGridState.create_from_state(state, logger)

    action_recipes = action_space.recipes
    recipes_mask = action_space.filter_recipes(action_recipes, behavioral_state)

    action_specs = [action_space.specify_goal(recipe, behavioral_state) if recipes_mask[i] else None
                    for i, recipe in enumerate(action_recipes)]

    spec_calm = [action_specs[i] for i, recipe in enumerate(action_space.recipes)
                 if recipe.relative_lane == RelativeLane.SAME_LANE and
                 recipe.relative_lon == RelativeLongitudinalPosition.FRONT and
                 recipe.aggressiveness == AggressivenessLevel.CALM]
    spec_strd = [action_specs[i] for i, recipe in enumerate(action_space.recipes)
                 if recipe.relative_lane == RelativeLane.SAME_LANE and
                 recipe.relative_lon == RelativeLongitudinalPosition.FRONT and
                 recipe.aggressiveness == AggressivenessLevel.STANDARD]
    spec_aggr = [action_specs[i] for i, recipe in enumerate(action_space.recipes)
                 if recipe.relative_lane == RelativeLane.SAME_LANE and
                 recipe.relative_lon == RelativeLongitudinalPosition.FRONT and
                 recipe.aggressiveness == AggressivenessLevel.AGGRESSIVE]
    T = []
    if len(spec_calm) > 0 and spec_calm[0] is not None:
        T.append(spec_calm[0].t)
    if len(spec_strd) > 0 and spec_strd[0] is not None:
        T.append(spec_strd[0].t)
    if len(spec_aggr) > 0 and spec_aggr[0] is not None:
        T.append(spec_aggr[0].t)

    assert len(T) > 0
