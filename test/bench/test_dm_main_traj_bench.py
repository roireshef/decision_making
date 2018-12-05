import time
import numpy as np

import pytest

from multiprocessing import Process
from unittest.mock import MagicMock

from common_data.interface.py.idl_generated_files.Rte_Types import LcmPerceivedSelfLocalization
from common_data.interface.py.pubsub.Rte_Types_pubsub_topics import PubSubMessageTypes, PERCEIVED_SELF_LOCALIZATION_LCM, \
    TRAJECTORY_PLAN
from common_data.src.communication.pubsub.pubsub_factory import create_pubsub
from decision_making.paths import Paths
from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.test.bench import dm_main_trajectory_bench

test_fixed_trajectory_file = Paths.get_resource_absolute_path_filename(
        'fixed_trajectory_files/test_trajectory.txt')


@pytest.mark.skip(reason="scenario is obsolete. this test needs to be rewritten")
def test_DMMainTraj_Bench_SingleLocalizationMessage_TrajectoryOutput():

    #Read first point in the test trajectory to accommodate trigger condition
    f = open(file=test_fixed_trajectory_file, mode='r')
    line = f.readline()
    line_arr = list(map(float, line.replace("\n", "").split(", ")))
    start_x, start_y = line_arr[0], line_arr[1]

    #Construct localization message
    localization_msg = LcmPerceivedSelfLocalization()
    localization_msg.location.x = start_x
    localization_msg.location.y = start_y

    #create pubsub and subscribe a magic mock to the perceived localization topic
    pubsub = create_pubsub(PubSubMessageTypes)
    receive_output_mock = MagicMock()
    pubsub.subscribe(TRAJECTORY_PLAN, receive_output_mock)

    #load dm_main_trajectory_bench with the test trajectory file and wait for it to load
    dm_main_process = Process(target=dm_main_trajectory_bench.main, name='traj_bench_test',
                              args=tuple([test_fixed_trajectory_file, None, NavigationPlanMsg(np.array([20]))]))
    dm_main_process.start()
    time.sleep(2)

    pubsub.publish(PERCEIVED_SELF_LOCALIZATION_LCM, localization_msg)
    time.sleep(2)
    dm_main_process.terminate()

    receive_output_mock.assert_called()



