from common_data.dds.python.Communication.ddspubsub import DdsPubSub
from rte.python.logger.AV_logger import AV_Logger
from abc import abstractmethod

class DmModule():

    def __init__(self, dds : DdsPubSub, logger):
        self.dds = dds
        self.logger = logger

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def periodic_action(self):
        pass


