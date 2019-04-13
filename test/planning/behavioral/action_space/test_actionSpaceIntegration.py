from decision_making.src.utils.map_utils import MapUtils
from decision_making.test.messages.static_scene_fixture import scene_static_pg_no_split
from logging import Logger
import numpy as np
import pickle

from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.scene.scene_static_model import SceneStaticModel
from decision_making.src.planning.behavioral.action_space.dynamic_action_space import DynamicActionSpace
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import RelativeLane, RelativeLongitudinalPosition
from decision_making.src.planning.behavioral.default_config import DEFAULT_DYNAMIC_RECIPE_FILTERING
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.state.state import ObjectSize, State, EgoState, DynamicObject


# test specify for dynamic action from a slightly unsafe position:
# when the distance from the target is just 2 seconds * target velocity, without adding the cars' sizes
def test_specifyGoal_slightlyUnsafeState_shouldSucceed(scene_static_pg_no_split):
    SceneStaticModel.get_instance().set_scene_static(scene_static_pg_no_split)

    logger = Logger("test_specifyDynamicAction")
    road_segment_id = MapUtils.get_road_segment_ids()[0]
    num_lanes = len(MapUtils.get_lanes_ids_from_road_segment_id(road_segment_id))
    lane_ordinal = num_lanes // 2
    lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_segment_id)[lane_ordinal]
    ego_lon = 400.

    lane_lat = 0
    size = ObjectSize(4, 2, 1)
    frenet = MapUtils.get_lane_frenet_frame(lane_id)

    predictor = RoadFollowingPredictor(logger)
    action_space = DynamicActionSpace(logger, predictor, filtering=DEFAULT_DYNAMIC_RECIPE_FILTERING)

    # verify the peak acceleration does not exceed the limit by calculating the average acceleration from 0 to 50 km/h
    ego_vel = 10
    ego_cstate = frenet.fstate_to_cstate(np.array([ego_lon, ego_vel, 0, lane_lat, 0, 0]))
    ego = EgoState.create_from_cartesian_state(obj_id=0, timestamp=0, cartesian_state=ego_cstate, size=size, confidence=0)

    obj_vel = 10
    obj_lon = ego_lon + 20
    obj_cstate = frenet.fstate_to_cstate(np.array([obj_lon, obj_vel, 0, lane_lat, 0, 0]))
    obj = DynamicObject.create_from_cartesian_state(obj_id=0, timestamp=0, cartesian_state=obj_cstate, size=size, confidence=0)

    state = State(False, None, [obj], ego)
    behavioral_state = BehavioralGridState.create_from_state(state, NavigationPlanMsg(np.array([20])), logger)

    action_recipes = action_space.recipes
    recipes_mask = action_space.filter_recipes(action_recipes, behavioral_state)

    front_recipes = [recipe for i, recipe in enumerate(action_space.recipes)
                     if recipe.relative_lane == RelativeLane.SAME_LANE and
                     recipe.relative_lon == RelativeLongitudinalPosition.FRONT and
                     recipes_mask[i]]

    # verify that there is at least one valid recipe for dynamic actions
    assert len(front_recipes) > 0
