import pytest
import numpy as np
import pickle
import os
from decision_making.src.global_constants import PG_SPLIT_PICKLE_FILE_NAME, PG_PICKLE_FILE_NAME, \
    TESTABLE_MAP_PICKLE_FILE_NAME, SHORT_MAP_PICKLE_FILE_NAME
from decision_making.paths import Paths

from decision_making.src.messages.scene_common_messages import Header, MapOrigin, Timestamp
from decision_making.src.messages.scene_static_message import SceneStatic, DataSceneStatic, SceneRoadSegment, \
    MapRoadSegmentType, SceneLaneSegment, MapLaneType, LaneSegmentConnectivity, ManeuverType, NominalPathPoint, \
    MapLaneMarkerType, BoundaryPoint, AdjacentLane, MovingDirection
from decision_making.src.planning.types import FP_SX, FP_DX
from decision_making.src.scene.scene_static_model import SceneStaticModel

from mapping.src.exceptions import NextRoadNotFound


@pytest.fixture
def scene_static_pg_no_split():
    return pickle.load(open(Paths.get_map_absolute_path_filename(PG_PICKLE_FILE_NAME), 'rb'))


@pytest.fixture
def scene_static_pg_split():
    return pickle.load(open(Paths.get_map_absolute_path_filename(PG_SPLIT_PICKLE_FILE_NAME), 'rb'))


@pytest.fixture
def scene_static_testable():
    return pickle.load(open(Paths.get_map_absolute_path_filename(TESTABLE_MAP_PICKLE_FILE_NAME), 'rb'))


@pytest.fixture
def scene_static_short_map():
    return pickle.load(open(Paths.get_map_absolute_path_filename(SHORT_MAP_PICKLE_FILE_NAME), 'rb'))
