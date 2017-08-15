from common_data.dds.python.Communication.ddspubsub import DdsPubSub
from rte.python.logger.AV_logger import AV_Logger
from abc import abstractmethod

class DM_Module():

    def __init__(self, DDS : DdsPubSub, logger: AV_Logger):
        self.DDS = DDS
        self.logger = logger

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass


