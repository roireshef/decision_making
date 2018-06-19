from logging import Logger

from decision_making.src.planning.behavioral.action_space.static_action_space import StaticActionSpace
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState, RelativeLane
from decision_making.src.planning.behavioral.data_objects import AggressivenessLevel
from decision_making.src.planning.behavioral.default_config import DEFAULT_STATIC_RECIPE_FILTERING
from decision_making.src.state.state import ObjectSize, State, NewEgoState
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
    ego = NewEgoState.create_from_cartesian_state(obj_id=0, timestamp=0,
                                                  cartesian_state=[ego_cpoint[0], ego_cpoint[1], ego_yaw, ego_vel, 0, 0],
                                                  size=size, confidence=0)

    state = State(None, [], ego)
    behavioral_state = BehavioralGridState.create_from_state(state, logger)

    action_specs = action_space.specify_goals(action_space.recipes, behavioral_state)

    specs = [action_specs[i] for i, recipe in enumerate(action_space.recipes)
             if recipe.relative_lane == RelativeLane.SAME_LANE and recipe.aggressiveness == AggressivenessLevel.CALM
             and recipe.velocity == target_vel]

    # check specification of CALM SAME_LANE static action
    assert len(specs) > 0 and specs[0] is not None

