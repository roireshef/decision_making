from common_data.dds.python.Communication.ddspubsub import DdsPubSub
from rte.python.logger.AV_logger import AV_Logger
from abc import abstractmethod


class DmModule:
    def __init__(self, dds: DdsPubSub, logger: AV_Logger):
        self.logger.info("initializing module: " + self.__class__.__name__)
        self.dds = dds
        self.logger = logger

    @abstractmethod
    def _start_impl(self):
        pass

    def start(self):
        self.logger.info("starting module: " + self.__class__.__name__)
        self._start_impl()

    @abstractmethod
    def _stop_impl(self):
        pass

    def stop(self):
        self.logger.info("stopping module: " + self.__class__.__name__)
        self._stop_impl()

    @abstractmethod
    def _periodic_action_impl(self):
        pass

    def periodic_action(self):
        self.logger.info("executing periodic action at module: " + self.__class__.__name__)
        self._stop_impl()


