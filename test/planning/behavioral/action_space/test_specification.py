from logging import Logger
import numpy as np
import pytest

from decision_making.src.global_constants import LON_ACC_LIMITS
from decision_making.src.planning.behavioral.action_space.static_action_space import StaticActionSpace
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState, RelativeLane
from decision_making.src.planning.behavioral.data_objects import AggressivenessLevel
from decision_making.src.state.state import ObjectSize, EgoState, State
from mapping.src.service.map_service import MapService


def test_specifyStaticAction_veryCloseToTargetVelocity_shouldNotFail():

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


def test_specifyStaticAction_T_shouldBeReasonable():

    logger = Logger("test_specifyStaticAction")
    road_id = 20
    ego_lon = 400.
    lane_width = MapService.get_instance().get_road(road_id).lane_width
    road_mid_lat = MapService.get_instance().get_road(road_id).lanes_num * lane_width / 2
    size = ObjectSize(4, 2, 1)
    target_vel = 50./3.6

    action_space = StaticActionSpace(logger)

    # verify the peak acceleration does not exceed the limit by calculating the average acceleration from 0 to 50 km/h
    ego_vel = 0
    ego_cpoint, ego_yaw = MapService.get_instance().convert_road_to_global_coordinates(road_id, ego_lon, road_mid_lat - lane_width)
    ego = EgoState(0, 0, ego_cpoint[0], ego_cpoint[1], ego_cpoint[2], ego_yaw, size, 0, ego_vel, 0, 0, 0, 0)

    state = State(None, [], ego)
    behavioral_state = BehavioralGridState.create_from_state(state, logger)

    action_specs = [action_space.specify_goal(recipe, behavioral_state, i) for i, recipe in enumerate(action_space.recipes)]

    spec1_calm = [action_specs[i] for i, recipe in enumerate(action_space.recipes)
                  if recipe.relative_lane == RelativeLane.SAME_LANE and recipe.velocity <= target_vel and
                  recipe.aggressiveness==AggressivenessLevel.CALM]
    spec1_strd = [action_specs[i] for i, recipe in enumerate(action_space.recipes)
                  if recipe.relative_lane == RelativeLane.SAME_LANE and recipe.velocity <= target_vel and
                  recipe.aggressiveness==AggressivenessLevel.STANDARD]
    spec1_aggr = [action_specs[i] for i, recipe in enumerate(action_space.recipes)
                  if recipe.relative_lane == RelativeLane.SAME_LANE and recipe.velocity <= target_vel and
                  recipe.aggressiveness==AggressivenessLevel.AGGRESSIVE]
    assert spec1_aggr[0].t > (target_vel - ego_vel) / LON_ACC_LIMITS[1] + 2

    # verify the specification times are not too long for accelerations from 32 to 50 km/h
    ego_vel = 9
    ego_cpoint, ego_yaw = MapService.get_instance().convert_road_to_global_coordinates(road_id, ego_lon, road_mid_lat - lane_width)
    ego = EgoState(0, 0, ego_cpoint[0], ego_cpoint[1], ego_cpoint[2], ego_yaw, size, 0, ego_vel, 0, 0, 0, 0)

    state = State(None, [], ego)
    behavioral_state = BehavioralGridState.create_from_state(state, logger)

    action_specs = [action_space.specify_goal(recipe, behavioral_state, i) for i, recipe in enumerate(action_space.recipes)]

    spec2_calm = [action_specs[i] for i, recipe in enumerate(action_space.recipes)
                  if recipe.relative_lane == RelativeLane.SAME_LANE and recipe.velocity <= target_vel and
                  recipe.aggressiveness==AggressivenessLevel.CALM]
    spec2_strd = [action_specs[i] for i, recipe in enumerate(action_space.recipes)
                  if recipe.relative_lane == RelativeLane.SAME_LANE and recipe.velocity <= target_vel and
                  recipe.aggressiveness==AggressivenessLevel.STANDARD]
    spec2_aggr = [action_specs[i] for i, recipe in enumerate(action_space.recipes)
                  if recipe.relative_lane == RelativeLane.SAME_LANE and recipe.velocity <= target_vel and
                  recipe.aggressiveness==AggressivenessLevel.AGGRESSIVE]

    very_calm_acceleration = 1.    # 0-100 km/h in 28 sec
    standard_acceleration = 1.5    # 0-100 km/h in 18 sec
    aggressive_acceleration = 2.3  # 0-100 km/h in 12 sec
    assert spec2_calm[0].t < (target_vel - ego_vel) / very_calm_acceleration + 2
    assert spec2_strd[0].t < (target_vel - ego_vel) / standard_acceleration + 1.5
    assert spec2_aggr[0].t < (target_vel - ego_vel) / aggressive_acceleration + 1
