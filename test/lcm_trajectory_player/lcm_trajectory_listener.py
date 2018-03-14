from logging import Logger

from common_data.lcm.python.Communication.lcmpubsub import LcmPubSub
from common_data.src.communication.pubsub.pubsub import PubSub
from common_data.src.communication.pubsub.pubsub_factory import create_pubsub
from decision_making.src.infra.dm_module import DmModule
import time
import numpy as np

from decision_making.src.messages.trajectory_plan_message import TrajectoryPlanMsg
from rte.python.logger.AV_logger import AV_Logger

TRAJECTORY_TOPIC = "TRAJECTORY_LCM"
LCM_SOCKET_CONFIG = 'lcm_socket_config.json'

class traj_listener(DmModule):
    def __init__(self, pubsub: PubSub, logger: Logger):
        super().__init__(pubsub, logger)

    def _stop_impl(self):
        pass

    def _start_impl(self):
        self.pubsub.subscribe(TRAJECTORY_TOPIC, self._print_trajectory_msg)

    def _periodic_action_impl(self):
        pass

    def _print_trajectory_msg(self, traj_msg: TrajectoryPlanMsg):
        print(traj_msg.trajectory.shape)

def main():
    pubsub = create_pubsub(LCM_SOCKET_CONFIG, LcmPubSub)
    logger = AV_Logger.get_logger("bla")
    listener = traj_listener(pubsub, logger)
    listener._start_impl()

if __name__ == '__main__':
    main()
