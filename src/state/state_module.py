from decision_making.src.infra.dm_module import DM_Module
from rte.python.periodic_timer.periodic_timer import PeriodicTimer

import time
class state_module(DM_Module):

    def __init__(self, DDS, logger):
        super().__init__(DDS, logger)

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




