import time
from common_data.dds.python.Communication.ddspubsub import DdsPubSub
from decision_making.src.infra.dm_module import DM_Module
from decision_making.src.state.enriched_state import *
from rte.python.logger.AV_logger import AV_Logger
from rte.python.periodic_timer.periodic_timer import PeriodicTimer

class StateModule(DM_Module):

    def __init__(self, dds: DdsPubSub, logger):
        super().__init__(dds, logger)

    def start(self):
        self.logger.info("Starting state module")
        self.DDS.subscribe("StateSubscriber::DynamicObjectsReader", self.__dynamic_obj_callback)
        self.timer = PeriodicTimer(2, self.__timer_callback)
        self.timer.start()

    def stop(self):
        self.logger.info("Stopping state module")
        self.DDS.unsubscribe("StateSubscriber::DynamicObjectsReader")
        self.timer.stop()

    def __timer_callback(self):
        self.logger.info("periodic timer")

    def __dynamic_obj_callback(self, objects: dict):
        self.logger.info("got dynamic objects %s", objects)


