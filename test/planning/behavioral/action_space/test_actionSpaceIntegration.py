from logging import Logger

from decision_making.src.planning.behavioral.action_space.dynamic_action_space import DynamicActionSpace
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState, RelativeLane, \
    RelativeLongitudinalPosition
from decision_making.src.planning.behavioral.default_config import DEFAULT_DYNAMIC_RECIPE_FILTERING
from decision_making.src.prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.state.state import ObjectSize, EgoState, State
from mapping.src.service.map_service import MapService


# test specify for dynamic action from a slightly unsafe position:
# when the distance from the target is just 2 seconds * target velocity, without adding the cars' sizes
def test_specifyGoal_slightlyUnsafeState_shouldSucceed():
    logger = Logger("test_specifyDynamicAction")
    road_id = 20
    ego_lon = 400.
    lane_width = MapService.get_instance().get_road(road_id).lane_width
    road_mid_lat = MapService.get_instance().get_road(road_id).lanes_num * lane_width / 2
    size = ObjectSize(4, 2, 1)

    predictor = RoadFollowingPredictor(logger)
    action_space = DynamicActionSpace(logger, predictor, filtering=DEFAULT_DYNAMIC_RECIPE_FILTERING)

    # verify the peak acceleration does not exceed the limit by calculating the average acceleration from 0 to 50 km/h
    ego_vel = 10
    ego_cpoint, ego_yaw = MapService.get_instance().convert_road_to_global_coordinates(road_id, ego_lon,
                                                                                       road_mid_lat - lane_width)
    ego = EgoState(0, 0, ego_cpoint[0], ego_cpoint[1], ego_cpoint[2], ego_yaw, size, 0, ego_vel, 0, 0, 0)

    obj_vel = 10
    obj_lon = ego_lon + 20
    obj_cpoint, obj_yaw = MapService.get_instance().convert_road_to_global_coordinates(road_id, obj_lon,
                                                                                       road_mid_lat - lane_width)
    obj = EgoState(0, 0, obj_cpoint[0], obj_cpoint[1], obj_cpoint[2], obj_yaw, size, 0, obj_vel, 0, 0, 0)

    state = State(None, [obj], ego)
    behavioral_state = BehavioralGridState.create_from_state(state, logger)

    action_recipes = action_space.recipes
    recipes_mask = action_space.filter_recipes(action_recipes, behavioral_state)

    front_recipes = [recipe for i, recipe in enumerate(action_space.recipes)
                     if recipe.relative_lane == RelativeLane.SAME_LANE and
                     recipe.relative_lon == RelativeLongitudinalPosition.FRONT and
                     recipes_mask[i]]

    # verify that there is at least one valid recipe for dynamic actions
    assert len(front_recipes) > 0