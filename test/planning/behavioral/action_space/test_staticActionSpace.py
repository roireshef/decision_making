from decision_making.test.planning.behavioral.behavioral_state_fixtures import route_plan_20_30
from logging import Logger

import numpy as np
from decision_making.src.scene.scene_static_model import SceneStaticModel
from decision_making.src.planning.behavioral.action_space.static_action_space import StaticActionSpace
from decision_making.src.planning.behavioral.state.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import AggressivenessLevel, RelativeLane, LaneChangeInfo
from decision_making.src.planning.behavioral.default_config import DEFAULT_STATIC_RECIPE_FILTERING
from decision_making.src.state.state import ObjectSize, State, EgoState
from decision_making.src.utils.map_utils import MapUtils

from decision_making.test.messages.scene_static_fixture import scene_static_pg_split


# test Specify, when ego starts with velocity very close to the target velocity
# scene_static is a multi-segment map
def test_specifyGoals_closeToTargetVelocity_specifyNotFail(scene_static_pg_split, route_plan_20_30):

    scene_static_message = scene_static_pg_split
    for lane_segment in scene_static_message.s_Data.s_SceneStaticBase.as_scene_lane_segments:
        lane_segment.as_traffic_control_bar = []
    scene_static_message.s_Data.s_SceneStaticBase.as_scene_lane_segments[0].as_traffic_control_bar = []
    scene_static_message.s_Data.s_SceneStaticBase.as_static_traffic_control_device = []
    scene_static_message.s_Data.s_SceneStaticBase.as_dynamic_traffic_control_device = []

    SceneStaticModel.get_instance().set_scene_static(scene_static_message)

    logger = Logger("test_specifyStaticAction")
    road_segment_id = 21
    ego_lon = 120.
    lane_ids = MapUtils.get_lanes_ids_from_road_segment_id(road_segment_id)
    lane_id = lane_ids[int(len(lane_ids)/2)]
    size = ObjectSize(4, 2, 1)

    action_space = StaticActionSpace(logger, DEFAULT_STATIC_RECIPE_FILTERING)

    target_vel = action_space.recipes[0].velocity
    ego_vel = target_vel + 0.01
    cstate = MapUtils.get_lane_frenet_frame(lane_id).fstate_to_cstate(np.array([ego_lon, ego_vel, 0, 0, 0, 0]))

    ego = EgoState.create_from_cartesian_state(obj_id=0, timestamp=0, cartesian_state=cstate, size=size, confidence=0,
                                               off_map=False, lane_change_info = LaneChangeInfo(None, np.array([]), False, 0.0))

    state = State(False, None, [], ego)
    behavioral_state = BehavioralGridState.create_from_state(state, route_plan_20_30, logger)
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

