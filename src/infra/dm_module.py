from abc import abstractmethod, ABCMeta
from logging import Logger

import six

from common_data.dds.python.Communication.ddspubsub import DdsPubSub


@six.add_metaclass(ABCMeta)
class DmModule:
    """
    Abstract class which is implemented in functional DM modules and facades.
    """
    def __init__(self, dds: DdsPubSub, logger: Logger) -> None:
        """
        :param dds: Inter-process communication interface.
        :param logger: Logging interface.
        """
        self.dds = dds
        self.logger = logger
        self.logger.info("initializing module: " + self.__class__.__name__)

    @abstractmethod
    def _start_impl(self) -> None:
        """
        Implementation specific start script.
        """
        pass

    def start(self) -> None:
        self.logger.info("starting module: " + self.__class__.__name__)
        self._start_impl()

    @abstractmethod
    def _stop_impl(self) -> None:
        """
        Implementation specific stop script.
        """
        pass

    def stop(self) -> None:
        self.logger.info("stopping module: " + self.__class__.__name__)
        self._stop_impl()

    @abstractmethod
    def _periodic_action_impl(self) -> None:
        """
        Implementation specific script for execution upon event.
        """
        pass

    def periodic_action(self) -> None:
        """
        Perform triggered action and write logging messages.
        """
        self.logger.debug("executing periodic action at module: " + self.__class__.__name__)
        self._periodic_action_impl()
        self.logger.debug("finished periodic action at module: " + self.__class__.__name__)
