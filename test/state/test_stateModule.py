from decision_making.src.global_constants import STATE_MODULE_NAME_FOR_LOGGING
from decision_making.src.state.state import OccupancyState, EgoState
from decision_making.src.state.state_module import StateModule
from decision_making.test.dds.mock_ddspubsub import DdsPubSubMock
from mapping.src.model.map_api import MapAPI
from rte.python.logger.AV_logger import AV_Logger
from spcog.decision_making_sim.test.fixtures import dynamic_objects_not_in_fov, dynamic_objects_in_fov
from mapping.test.model.testable_map_fixtures import testable_map_api
from decision_making.test.planning.custom_fixtures import dds_pubsub, ego_state_fix
import numpy as np


def test_dynamicObjCallback_objectInAndOutOfFOV_stateWithInFOVObject(dds_pubsub : DdsPubSubMock, dynamic_objects_in_fov : dict,
                                                                     dynamic_objects_not_in_fov : dict, ego_state_fix : EgoState,
                                                                     testable_map_api : MapAPI):
    """

    :param dds_pubsub: Inter-process communication interface.
    :param dynamic_objects_in_fov: Fixture of a serialized dynamic object data located within the field of view
            (FOV).
    :param dynamic_objects_not_in_fov: Fixture of a serialized dynamic object with the same id as above but now it's located
                                        out of the field of view.
    :param ego_state_fix: Fixture of an ego state compatible with the above two fixtures.
    :param testable_map_api: An interface to a test map.

    This test checks the memory functionality of the StateModule. It initially sends into the StateModule a dynamic object
    in the FOV followed by a message indicating that the object is out of FOV. The test then asserts that the last known
    object properties have been "remembered".
    """
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
    assert new_dyn_obj_list[0].v_x == dynamic_objects_in_fov["dynamic_objects"][0]["velocity"]["v_x"]
    assert new_dyn_obj_list[0].v_y == dynamic_objects_in_fov["dynamic_objects"][0]["velocity"]["v_y"]



    state_module.stop()