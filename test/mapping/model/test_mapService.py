import pytest

from decision_making.src.mapping.service.map_service import MapService


def test_mapServiceInitialization_():
    """
    Tests the MapService initialization
    :return:
    """
    MapService.initialize()
    map_api = MapService.get_instance()
    assert map_api is not None


def test_mapServiceInitialization_():
    """
    Tests the MapService initialization with the get_instance() function
    :return:
    """
    map_api = MapService.get_instance()
    assert map_api is not None
