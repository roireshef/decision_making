from abc import abstractmethod, ABCMeta
from rte.python.logger.AV_logger import AV_Logger, INFO

from common_data.src.communication.pubsub.pubsub import PubSub
import six
import rte.python.profiler as prof


@six.add_metaclass(ABCMeta)
class DmModule:
    """
    Abstract class which is implemented in functional DM modules and facades.
    """
    def __init__(self, pubsub, logger):
        # type: (PubSub, AV_Logger) -> None
        self.pubsub = pubsub
        """
        :param dds: Inter-process communication interface.
        :param logger: Logging interface.
        """
        self.name = self.__class__.__name__
        self.logger = logger
        self.logger.info("initializing module: %s", self.name)
        self.profiler_log_level = INFO

    @abstractmethod
    def _start_impl(self):
        # type: () -> None
        """
        Implementation specific start script.
        """
        pass

    def start(self):
        # type: () -> None
        self.logger.info("starting module: %s", self.name)
        self._start_impl()

    @abstractmethod
    def _stop_impl(self):
        # type: () -> None
        """
        Implementation specific stop script.
        """
        pass

    def stop(self):
        # type: () -> None
        self.logger.info("stopping module: %s", self.name)
        self._stop_impl()

    @abstractmethod
    def _periodic_action_impl(self):
        # type: () -> None
        """
        Implementation specific script for execution upon event.
        """
        pass

    def periodic_action(self):
        # type: () -> None
        """
        Perform triggered action and write logging messages.
        """
        self.logger.debug("executing periodic action at module: %s", self.name)
        with prof.time_range("%s:periodic", self.name, log_level=self.profiler_log_level):
            self._periodic_action_impl()
        self.logger.debug("finished periodic action at module: %s", self.name)

