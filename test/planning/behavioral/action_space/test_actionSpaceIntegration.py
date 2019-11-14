from decision_making.src.utils.map_utils import MapUtils
from logging import Logger
import numpy as np

from decision_making.src.scene.scene_static_model import SceneStaticModel
from decision_making.src.planning.behavioral.action_space.dynamic_action_space import DynamicActionSpace
from decision_making.src.planning.behavioral.state.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import RelativeLane, RelativeLongitudinalPosition
from decision_making.src.planning.behavioral.default_config import DEFAULT_DYNAMIC_RECIPE_FILTERING
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.state.state import ObjectSize, State, EgoState, DynamicObject

from decision_making.test.messages.scene_static_fixture import scene_static_pg_no_split
from decision_making.test.planning.behavioral.behavioral_state_fixtures import route_plan_20
from decision_making.test.planning.custom_fixtures import turn_signal

# test specify for dynamic action from a slightly unsafe position:
# when the distance from the target is just 2 seconds * target velocity, without adding the cars' sizes


def test_specifyGoal_slightlyUnsafeState_shouldSucceed(scene_static_pg_no_split, route_plan_20, turn_signal):
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
    ego = EgoState.create_from_cartesian_state(obj_id=0, timestamp=0, cartesian_state=ego_cstate, size=size,
                                               confidence=0, off_map=False)

    obj_vel = 10
    obj_lon = ego_lon + 20
    obj_cstate = frenet.fstate_to_cstate(np.array([obj_lon, obj_vel, 0, lane_lat, 0, 0]))
    obj = DynamicObject.create_from_cartesian_state(obj_id=0, timestamp=0, cartesian_state=obj_cstate, size=size,
                                                    confidence=0, off_map=False)

    state = State(False, None, [obj], ego)
    behavioral_state = BehavioralGridState.create_from_state(state, route_plan_20, turn_signal, logger)

    action_recipes = action_space.recipes
    recipes_mask = action_space.filter_recipes(action_recipes, behavioral_state)

    front_recipes = [recipe for i, recipe in enumerate(action_space.recipes)
                     if recipe.relative_lane == RelativeLane.SAME_LANE and
                     recipe.relative_lon == RelativeLongitudinalPosition.FRONT and
                     recipes_mask[i]]

    # verify that there is at least one valid recipe for dynamic actions
    assert len(front_recipes) > 0
