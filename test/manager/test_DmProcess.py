import os

import pytest

from decision_making.src.global_constants import BEHAVIORAL_PLANNING_NAME_FOR_LOGGING
from decision_making.src.manager.dm_process import DmProcess
from decision_making.src.manager.dm_trigger import DmTriggerType
from decision_making.src.planning.behavioral.behavioral_planning_facade import BehavioralPlanningFacade
from decision_making.test.constants import LCM_PUB_SUB_MOCK_NAME_FOR_LOGGING
from decision_making.test.planning.behavioral.mock_behavioral_facade import BehavioralFacadeMock
from decision_making.test.pubsub.mock_pubsub import PubSubMock
from rte.python.logger.AV_logger import AV_Logger
from decision_making.test.planning import custom_fixtures


def create_behavioral_planner() -> BehavioralPlanningFacade:

    pubsub = PubSubMock(logger=AV_Logger.get_logger(LCM_PUB_SUB_MOCK_NAME_FOR_LOGGING))

    logger = AV_Logger.get_logger(BEHAVIORAL_PLANNING_NAME_FOR_LOGGING)

    behavioral_module = BehavioralFacadeMock(pubsub=pubsub, logger=logger, trigger_pos=None, trajectory_params=None,
                                             visualization_msg=None)
    return behavioral_module


@pytest.fixture()
def dm_process():
    # Initializations

    dm_process_behavioral = DmProcess(create_behavioral_planner, trigger_type=DmTriggerType.DM_TRIGGER_NONE,
                                      trigger_args={}, name='BP')

    dm_process_behavioral.start_process()

    yield dm_process_behavioral

    print("tear down")
    dm_process_behavioral.stop_process()


# Integration test
def test_StartProcess_SanityCheck_ValidResult(dm_process):

    expected_module_name = 'BP'

    assert dm_process.name == expected_module_name
    assert dm_process.process.is_alive()
    assert os.getpid != dm_process.process.pid
