from logging import Logger
import numpy as np

from decision_making.src.scene.scene_static_model import SceneStaticModel
from decision_making.src.planning.behavioral.action_space.dynamic_action_space import DynamicActionSpace
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import RelativeLane, RelativeLongitudinalPosition
from decision_making.src.planning.behavioral.default_config import DEFAULT_DYNAMIC_RECIPE_FILTERING
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.state.state import ObjectSize, State, EgoState, DynamicObject
from mapping.src.service.map_service import MapService

from decision_making.test.messages.static_scene_fixture import scene_static_no_split
from decision_making.test.planning.behavioral.behavioral_state_fixtures import route_plan_20

# test specify for dynamic action from a slightly unsafe position:
# when the distance from the target is just 2 seconds * target velocity, without adding the cars' sizes


def test_specifyGoal_slightlyUnsafeState_shouldSucceed(scene_static_no_split, route_plan_20):
    SceneStaticModel.get_instance().set_scene_static(scene_static_no_split)

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
    ego = EgoState.create_from_cartesian_state(obj_id=0, timestamp=0,
                                               cartesian_state=np.array([ego_cpoint[0], ego_cpoint[1], ego_yaw, ego_vel, 0, 0]),
                                               size=size, confidence=0)

    obj_vel = 10
    obj_lon = ego_lon + 20
    obj_cpoint, obj_yaw = MapService.get_instance().convert_road_to_global_coordinates(road_id, obj_lon,
                                                                                       road_mid_lat - lane_width)
    obj = DynamicObject.create_from_cartesian_state(obj_id=0, timestamp=0,
                                                    cartesian_state=np.array([obj_cpoint[0], obj_cpoint[1], obj_yaw, obj_vel, 0.0, 0.0]),
                                                    size=size, confidence=0)

    state = State(None, [obj], ego)
    behavioral_state = BehavioralGridState.create_from_state(state, route_plan_20, logger)

    action_recipes = action_space.recipes
    recipes_mask = action_space.filter_recipes(action_recipes, behavioral_state)

    front_recipes = [recipe for i, recipe in enumerate(action_space.recipes)
                     if recipe.relative_lane == RelativeLane.SAME_LANE and
                     recipe.relative_lon == RelativeLongitudinalPosition.FRONT and
                     recipes_mask[i]]

    # verify that there is at least one valid recipe for dynamic actions
    assert len(front_recipes) > 0
