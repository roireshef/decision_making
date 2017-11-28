from common_data.dds.python.Communication.ddspubsub import DdsPubSub
from decision_making.src.global_constants import STATE_MODULE_NAME_FOR_LOGGING
from decision_making.src.state.state import OccupancyState
from decision_making.src.state.state_module import StateModule
from rte.python.logger.AV_logger import AV_Logger
from spcog.decision_making_sim.test.fixtures import dynamic_objects_not_in_fov, dynamic_objects_in_fov
from mapping.test.model.testable_map_fixtures import testable_map_api
from decision_making.test.planning.custom_fixtures import dds_pubsub, ego_state_fix
import numpy as np


def test_dynamicObjCallback_objectInAndOutOfFOV_stateWithInFOVObject(dds_pubsub, dynamic_objects_in_fov,
                                                                     dynamic_objects_not_in_fov, ego_state_fix,
                                                                     testable_map_api):

    logger = AV_Logger.get_logger(STATE_MODULE_NAME_FOR_LOGGING)

    state_module = StateModule(dds=dds_pubsub, logger=logger, map_api=testable_map_api,
                               occupancy_state=OccupancyState(0, np.array([]), np.array([])),
                               dynamic_objects=None, ego_state=ego_state_fix)
    state_module.start()

    state_module.create_dyn_obj_list(dynamic_objects_in_fov)
    new_dyn_obj_list = state_module.create_dyn_obj_list(dynamic_objects_not_in_fov)
    assert len(new_dyn_obj_list) == 1
    assert new_dyn_obj_list[0].timestamp == dynamic_objects_in_fov["timestamp"]
    assert new_dyn_obj_list[0].x == dynamic_objects_in_fov["dynamic_objects"][0]["location"]["x"]
    assert new_dyn_obj_list[0].y == dynamic_objects_in_fov["dynamic_objects"][0]["location"]["y"]
    glob_v_x = dynamic_objects_in_fov["dynamic_objects"][0]["velocity"]["v_x"]
    glob_v_y = dynamic_objects_in_fov["dynamic_objects"][0]["velocity"]["v_y"]
    yaw = dynamic_objects_in_fov["dynamic_objects"][0]["bbox"]["yaw"]

    # convert velocity from map coordinates to relative to its own yaw
    v_x = np.cos(yaw) * glob_v_x + np.sin(yaw) * glob_v_y
    v_y = -np.sin(yaw) * glob_v_x + np.cos(yaw) * glob_v_y

    assert new_dyn_obj_list[0].v_x == v_x
    assert new_dyn_obj_list[0].v_y == v_y



    state_module.stop()