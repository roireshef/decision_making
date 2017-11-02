from decision_making.src.global_constants import BEHAVIORAL_PLANNING_NAME_FOR_LOGGING
from decision_making.src.planning.behavioral.policies.default_policy import DefaultBehavioralState
from decision_making.test.planning.custom_fixtures import state_fix, navigation_plan
from mapping.test.model.testable_map_fixtures import testable_map_api
from rte.python.logger.AV_logger import AV_Logger
import pytest

@pytest.fixture(scope='function')
def default_policy_behavioral_state(navigation_plan, testable_map_api, state_fix):
    logger = AV_Logger.get_logger(BEHAVIORAL_PLANNING_NAME_FOR_LOGGING)
    yield DefaultBehavioralState(logger=logger, map_api=testable_map_api, navigation_plan=navigation_plan,
                                 ego_state=state_fix.ego_state, dynamic_objects_on_road=state_fix.dynamic_objects,
                                 road_semantic_occupancy_grid=None)

