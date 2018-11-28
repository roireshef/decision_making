from unittest.mock import patch

import pytest

from decision_making.src.mapping.scene_model import SceneModel
from decision_making.src.messages.scene_static_message import SceneStatic
from decision_making.src.state.state import DynamicObject
from decision_making.src.utils.map_utils import MapUtils
from decision_making.test.constants import MAP_SERVICE_ABSOLUTE_PATH
from decision_making.test.messages.static_scene_fixture import scene_static
from mapping.test.model.testable_map_fixtures import map_api_mock
from decision_making.test.planning.custom_fixtures import dyn_obj_outside_road, dyn_obj_on_road
from decision_making.test.messages.static_scene_fixture import scene_static

@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
@pytest.mark.skip('method removed on lane-based.')
def test_isObjectOnRoad_objectOffOfRoad_False(dyn_obj_outside_road: DynamicObject):
    """
    :param ego_state_fix: Fixture of an ego state.

    Checking functionality of _is_object_on_road for an object that is off the road.
    """

    actual_result = MapUtils.is_object_on_road(dyn_obj_outside_road.map_state)
    expected_result = False
    assert expected_result == actual_result


@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
@pytest.mark.skip('method removed on lane-based.')
def test_isObjectOnRoad_objectOnRoad_True(dyn_obj_on_road: DynamicObject):
    """
    :param pubsub: Inter-process communication interface.
    :param ego_state_fix: Fixture of an ego state.

    Checking functionality of _is_object_on_road for an object that is on the road.
    """

    actual_result = MapUtils.is_object_on_road(dyn_obj_on_road.map_state)
    expected_result = True
    assert expected_result == actual_result


def test_getRoadSegmentIdFromLaneId_simple(scene_static: SceneStatic):
    SceneModel.get_instance().add_scene_static(scene_static)
    lane_id = 20
    expected_result = 1
    actual_result = MapUtils.get_road_segment_id_from_lane_id(lane_id)
    assert actual_result == expected_result

