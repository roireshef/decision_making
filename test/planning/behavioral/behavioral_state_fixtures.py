from decision_making.src.planning.behavioral.policies.default_policy import DefaultBehavioralState
from decision_making.test.planning.custom_fixtures import *
from mapping.test.model.testable_map_fixtures import *
from rte.python.logger.AV_logger import AV_Logger


@pytest.fixture(scope='function')
def default_policy_behavioral_state(navigation_fixture, testable_map_api, state):
    logger = AV_Logger.get_logger(BEHAVIORAL_PLANNING_NAME_FOR_LOGGING)
    yield DefaultBehavioralState(logger=logger, map_api=testable_map_api, navigation_plan=navigation_fixture,
                                 ego_state=state.ego_state, dynamic_objects_on_road=state.dynamic_objects)
