from src.infra.dm_module import DM_Module

class state_module(DM_Module):

    def __init__(self, DDS, logger):
        super().__init__(DDS, logger)

    def start(self):
        self.logger.info("Starting state module")
        self.DDS.subscribe("StateSubscriber::DynamicObjectsReader", self.DynamicObj_callback)

    def stop(self):
        self.logger.info("Stopping state module")
        self.DDS.unsubscribe("StateSubscriber::DynamicObjectsReader")

    def DynamicObj_callback(self, dict):
        self.logger.info("got dynamic objects %s", dict)