from unittest.mock import patch

from decision_making.src.state.state import NewDynamicObject
from decision_making.src.utils.map_utils import MapUtils
from decision_making.test.constants import MAP_SERVICE_ABSOLUTE_PATH
from mapping.test.model.testable_map_fixtures import map_api_mock
from decision_making.test.planning.custom_fixtures import dyn_obj_outside_road, dyn_obj_on_road

@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
def test_isObjectOnRoad_objectOffOfRoad_False(dyn_obj_outside_road: NewDynamicObject):
    """
    :param ego_state_fix: Fixture of an ego state.

    Checking functionality of _is_object_on_road for an object that is off the road.
    """

    actual_result = MapUtils.is_object_on_road(dyn_obj_outside_road.map_state)
    expected_result = False
    assert expected_result == actual_result


@patch(target=MAP_SERVICE_ABSOLUTE_PATH, new=map_api_mock)
def test_isObjectOnRoad_objectOnRoad_True(dyn_obj_on_road: NewDynamicObject):
    """
    :param pubsub: Inter-process communication interface.
    :param ego_state_fix: Fixture of an ego state.

    Checking functionality of _is_object_on_road for an object that is on the road.
    """

    actual_result = MapUtils.is_object_on_road(dyn_obj_on_road.map_state)
    expected_result = True
    assert expected_result == actual_result