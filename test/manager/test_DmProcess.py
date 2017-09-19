import os

from common_data.dds.python.Communication.ddspubsub import DdsPubSub
from decision_making.src.global_constants import BEHAVIORAL_PLANNING_NAME_FOR_LOGGING, \
    BEHAVIORAL_PLANNER_DDS_PARTICIPANT, DECISION_MAKING_DDS_FILE
from decision_making.src.manager.dm_process import DmProcess
from decision_making.src.manager.dm_trigger import DmTriggerType
from decision_making.src.planning.behavioral.behavioral_facade import BehavioralFacade
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState
from decision_making.src.planning.behavioral.default_policy import DefaultPolicy
from decision_making.src.state.state import EgoState, RoadLocalization, ObjectSize
from rte.python.logger.AV_logger import AV_Logger

import pytest


def create_behavioral_planner() -> BehavioralFacade:
    logger = AV_Logger.get_logger(BEHAVIORAL_PLANNING_NAME_FOR_LOGGING)
    dds = DdsPubSub(BEHAVIORAL_PLANNER_DDS_PARTICIPANT, DECISION_MAKING_DDS_FILE)
    # TODO: fill the policy
    policy_params = dict()
    policy = DefaultPolicy(policy_params)
    size = ObjectSize(0, 0, 0)
    road_localization = RoadLocalization(0, 0, 0, 0, 0, 0)
    ego_state = EgoState(0, 0, 0, 0, 0, 0, size, 0, 0, 0, 0, 0, 0, road_localization)
    behavioral_state = BehavioralState(ego_state=ego_state)
    return BehavioralFacade(dds=dds, logger=logger, policy=policy, behavioral_state=behavioral_state)


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
