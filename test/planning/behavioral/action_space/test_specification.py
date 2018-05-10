from logging import Logger
import numpy as np
import pytest

from decision_making.src.planning.behavioral.action_space.static_action_space import StaticActionSpace
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.state.state import ObjectSize, EgoState, State
from mapping.src.service.map_service import MapService


def test_specifyStaticAction_veryCloseToTargetVelocity_shouldNotFail():

    logger = Logger("test_specifyStaticAction")
    road_id = 20
    ego_lon = 400.
    lane_width = MapService.get_instance().get_road(road_id).lane_width
    road_mid_lat = MapService.get_instance().get_road(road_id).lanes_num * lane_width / 2
    size = ObjectSize(4, 2, 1)

    action_space = StaticActionSpace(logger)

    ego_vel = action_space.recipes[0].velocity + 0.1
    ego_cpoint, ego_yaw = MapService.get_instance().convert_road_to_global_coordinates(road_id, ego_lon, road_mid_lat - lane_width)
    ego = EgoState(0, 0, ego_cpoint[0], ego_cpoint[1], ego_cpoint[2], ego_yaw, size, 0, ego_vel, 0, 0, 0, 0)

    state = State(None, [], ego)
    behavioral_state = BehavioralGridState.create_from_state(state, logger)

    action_specs = [action_space.specify_goal(recipe, behavioral_state) for i, recipe in enumerate(action_space.recipes)]

    # check specification of CALM SAME_LANE static action
    assert action_specs[15] is not None
