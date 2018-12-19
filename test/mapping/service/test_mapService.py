from decision_making.src.mapping.service.map_service import MapService
import os
import common_data as COMMON_DATA_ROOT


def test_mapServiceNotInitializated_InitializesAndReturnsValidInstance():
    """
    Tests the MapService initialization
    :return:
    """
    MapService.initialize()
    map_api = MapService.get_instance()
    assert map_api is not None
    assert map_api._cached_map_model is not None
    assert map_api._cached_map_model._frame_origin is not None
    assert len(map_api._cached_map_model._roads_data) == 1

    road = list(map_api._cached_map_model._roads_data.items())[0][1]
    assert len(road._longitudes) > 0
    assert len(road._points) > 0


def test_mapServiceNotInitializated_GetInstanceInitializesService():
    """
    Tests the MapService initialization
    :return:
    """
    map_api = MapService.get_instance()
    assert map_api is not None


def test_mapServiceLoadsCustomMap():
    """
    verify that map service can load a non-default map correctly
    :return:
    """
    MapService.initialize("OvalMilford.bin")
    map_api = MapService.get_instance()

    frame_origin = map_api.get_frame_origin()

    assert frame_origin == [42.571837133673476, -83.698580799484716], "incorrect frame origin"

    MapService.initialize()

    map_api_default = MapService.get_instance()

    frame_origin_default = map_api_default.get_frame_origin()

    assert frame_origin_default == [32.208524709999999, 34.835358479999996], "incorrect frame origin"





