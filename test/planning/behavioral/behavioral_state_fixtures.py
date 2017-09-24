from decision_making.src.global_constants import BEHAVIORAL_PLANNING_NAME_FOR_LOGGING
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState
from decision_making.src.planning.utils.geometry_utils import CartesianFrame
from decision_making.test.planning.custom_map_fixtures import *
from mapping.test.model.testable_map_fixtures import *
from rte.python.logger.AV_logger import AV_Logger


@pytest.fixture(scope='function')
def default_policy_behavioral_state(navigation_fixture, testable_map_api, state):
    logger = AV_Logger.get_logger(BEHAVIORAL_PLANNING_NAME_FOR_LOGGING)
    ego_orientation = np.array(CartesianFrame.convert_yaw_to_quaternion(state.ego_state.yaw))
    yield BehavioralState(logger=logger, map_api=testable_map_api, navigation_plan=navigation_fixture,
                          ego_state=state.ego_state, timestamp=state.ego_state.timestamp,
                          ego_position=np.array([0.0, 0.0, 0.0]), ego_orientation=ego_orientation,
                          ego_yaw=state.ego_state.yaw, ego_velocity=0.0, ego_road_id=1,
                          ego_on_road=True, dynamic_objects_on_road=state.dynamic_objects)
