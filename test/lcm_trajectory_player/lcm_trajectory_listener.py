from logging import Logger

from common_data.lcm.config import pubsub_topics
from common_data.lcm.generatedFiles.gm_lcm import LcmPerceivedDynamicObjectList, LcmPerceivedDynamicObject, \
    LcmPerceivedSelfLocalization
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
DYN_OBJECT_TOPIC = 'DYNAMIC_OBJECT_LIST_LCM'
SELF_LOC_TOPIC = 'SELF_LOCALIZATION_LCM'

class traj_listener(DmModule):
    def __init__(self, pubsub: PubSub, logger: Logger):
        super().__init__(pubsub, logger)
        self.last_dyn_obj_time = time.time()

    def _stop_impl(self):
        pass

    def _start_impl(self):
        self.pubsub.subscribe(TRAJECTORY_TOPIC, self._print_trajectory_msg)
        self.pubsub.subscribe(DYN_OBJECT_TOPIC, self._print_dynamic_objects)
        self.pubsub.subscribe(pubsub_topics.PERCEIVED_SELF_LOCALIZATION_TOPIC, self._print_self_localization)

    def _periodic_action_impl(self):
        pass

    def _print_trajectory_msg(self, traj_msg: TrajectoryPlanMsg):
        print(traj_msg.trajectory.shape)

    def _print_dynamic_objects(self, dyn_obj_list: LcmPerceivedDynamicObjectList):
        print('received dynamic objects. dt =  ' + str(time.time() - self.last_dyn_obj_time))
        self.last_dyn_obj_time = time.time()
        for dyn_obj in dyn_obj_list.dynamic_objects:
            dyn_obj: LcmPerceivedDynamicObject = dyn_obj
            print (dyn_obj.encode())

    def _print_self_localization(self, self_localization: LcmPerceivedSelfLocalization):
        print('received localization.')
        print(self_localization)



def main():
    pubsub = create_pubsub(LCM_SOCKET_CONFIG, LcmPubSub)
    logger = AV_Logger.get_logger("bla")
    listener = traj_listener(pubsub, logger)
    listener._start_impl()

if __name__ == '__main__':
    main()
