import os

import pytest

from decision_making.src.global_constants import BEHAVIORAL_PLANNING_NAME_FOR_LOGGING
from decision_making.src.manager.dm_process import DmProcess
from decision_making.src.manager.dm_trigger import DmTriggerType
from decision_making.src.planning.behavioral.behavioral_facade import BehavioralFacade
from decision_making.test.planning.behavioral.mock_behavioral_facade import BehavioralFacadeMock
from rte.python.logger.AV_logger import AV_Logger
from  decision_making.test.planning import custom_fixtures


def create_behavioral_planner() -> BehavioralFacade:

    dds_pubsub = custom_fixtures.dds_pubsub
    trajectory_params = custom_fixtures.trajectory_params
    behavioral_visualization_msg = custom_fixtures.behavioral_visualization_msg

    logger = AV_Logger.get_logger(BEHAVIORAL_PLANNING_NAME_FOR_LOGGING)

    behavioral_module = BehavioralFacadeMock(dds=dds_pubsub, logger=logger, trajectory_params=trajectory_params,
                                             visualization_msg=behavioral_visualization_msg)
    return behavioral_module

@pytest.fixture()
def dm_process():
    # Initializations

    dm_process_behavioral = DmProcess(create_behavioral_planner, trigger_type=DmTriggerType.DM_TRIGGER_NONE,
                                      trigger_args={})

    dm_process_behavioral.start_process()

    yield dm_process_behavioral

    print("tear down")
    dm_process_behavioral.stop_process()


# Integration test
def test_StartProcess_SanityCheck_ValidResult(dm_process):

    expected_module_name = "create_behavioral_planner"

    assert dm_process.name == expected_module_name
    assert dm_process.process.is_alive()
    assert os.getpid != dm_process.process.pid
