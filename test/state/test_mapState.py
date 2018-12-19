import numpy as np

from decision_making.src.mapping.model.constants import ROAD_SHOULDERS_WIDTH
from decision_making.src.scene.scene_static_model import SceneStaticModel
from decision_making.src.state.map_state import MapState
from decision_making.src.utils.map_utils import MapUtils
from decision_making.src.mapping.service.map_service import MapService
from decision_making.test.messages.static_scene_fixture import scene_static_no_split, scene_static, \
    create_scene_static_from_map_api


def test_isOnRoad_onRighestLane_validateOnRoad(scene_static_no_split):
    """
    validate that point inside the road both longitudinally and laterally is on road
    """
    SceneStaticModel.get_instance().set_scene_static(scene_static_no_split)
    road_ids = MapService.get_instance()._cached_map_model.get_road_ids()
    lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_ids[0])[0]
    map_state = MapState(lane_fstate=np.array([100, 0, 0, 1, 0, 0]), lane_id=lane_id)
    assert map_state.is_on_road()


def test_isOnRoad_outsideRoadLongitudinally_validateNotOnRoad(scene_static_no_split):
    """
    validate that point outside the road longitudinally is not on road
    """
    SceneStaticModel.get_instance().set_scene_static(scene_static_no_split)
    road_ids = MapService.get_instance()._cached_map_model.get_road_ids()
    lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_ids[0])[0]
    map_state = MapState(lane_fstate=np.array([2000, 0, 0, 0, 0, 0]), lane_id=lane_id)
    assert not map_state.is_on_road()


def test_isOnRoad_outsideRoadLaterally_validateNotOnRoad(scene_static_no_split):
    """
    validate that point outside the road laterally is not on road
    """
    SceneStaticModel.get_instance().set_scene_static(scene_static_no_split)
    road_ids = MapService.get_instance()._cached_map_model.get_road_ids()
    lane_id = MapUtils.get_lanes_ids_from_road_segment_id(road_ids[0])[0]
    lane_width = MapUtils.get_lane_width(lane_id, 0)
    map_state = MapState(lane_fstate=np.array([100, 0, 0, lane_width/2 + ROAD_SHOULDERS_WIDTH + 0.1, 0, 0]), lane_id=lane_id)
    assert not map_state.is_on_road()
