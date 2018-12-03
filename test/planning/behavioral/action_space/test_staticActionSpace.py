from logging import Logger

import numpy as np

from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.planning.behavioral.action_space.static_action_space import StaticActionSpace
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import AggressivenessLevel, RelativeLane
from decision_making.src.planning.behavioral.default_config import DEFAULT_STATIC_RECIPE_FILTERING
from decision_making.src.state.state import ObjectSize, State, EgoState
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
    ego = EgoState.create_from_cartesian_state(obj_id=0, timestamp=0,
                                               cartesian_state=np.array([ego_cpoint[0], ego_cpoint[1], ego_yaw, ego_vel, 0, 0]),
                                               size=size, confidence=0)

    state = State(None, [], ego)
    behavioral_state = BehavioralGridState.create_from_state(state, NavigationPlanMsg(np.array([20])), logger)
    # ego is located on the rightest lane, so filter recipes to the right
    filtered_recipes = [recipe for recipe in action_space.recipes if recipe.relative_lane != RelativeLane.RIGHT_LANE]

    action_specs = action_space.specify_goals(filtered_recipes, behavioral_state)

    # check specification of CALM SAME_LANE static action
    same_lane_specs = [action_specs[i] for i, recipe in enumerate(filtered_recipes)
                       if recipe.relative_lane == RelativeLane.SAME_LANE and recipe.aggressiveness == AggressivenessLevel.CALM
                       and recipe.velocity == target_vel]
    assert len(same_lane_specs) > 0 and same_lane_specs[0] is not None

    # check specification of CALM LEFT_LANE static action
    left_lane_specs = [action_specs[i] for i, recipe in enumerate(filtered_recipes)
                       if recipe.relative_lane == RelativeLane.LEFT_LANE and recipe.aggressiveness == AggressivenessLevel.CALM
                       and recipe.velocity == target_vel]
    assert len(left_lane_specs) > 0 and left_lane_specs[0] is not None

