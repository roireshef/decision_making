import time
from common_data.dds.python.Communication.ddspubsub import DdsPubSub
from decision_making.src.infra.dm_module import DmModule
from decision_making.src.state.enriched_state import *
from rte.python.logger.AV_logger import AV_Logger
from decision_making.src.global_constants import *

class StateModule(DmModule):

    def __init__(self, dds: DdsPubSub, logger):
        super().__init__(dds, logger)

    def start(self):
        self.dds.subscribe(DYNAMIC_OBJECTS_SUBSCRIBE_TOPIC, self.__dynamic_obj_callback)
        self.dds.subscribe(SELF_LOCALIZATION_SUBSCRIBE_TOPIC, self.__self_localization_callback)
        self.dds.subscribe(OCCUPANCY_STATE_SUBSCRIBE_TOPIC, self.__occupancy_state_callback)

    def stop(self):
        self.dds.unsubscribe(DYNAMIC_OBJECTS_SUBSCRIBE_TOPIC)
        self.dds.unsubscribe(SELF_LOCALIZATION_SUBSCRIBE_TOPIC)
        self.dds.unsubscribe(OCCUPANCY_STATE_SUBSCRIBE_TOPIC)

    def periodic_action(self):
        pass

    def __dynamic_obj_callback(self, objects: dict):
        self.logger.info("got dynamic objects %s", objects)

    def __self_localization_callback(self, ego_localization: dict):
        self.logger.debug("got self localization %s", ego_localization)

    def __occupancy_state_callback(self, occupancy: dict):
        self.logger.info("got occupancy status %s", occupancy)
