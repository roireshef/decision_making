from abc import abstractmethod, ABCMeta
from logging import Logger

from decision_making.src.infra.pubsub import PubSub
import six
import rte.python.profiler as prof
from decision_making.src.utils.dm_profiler import DMProfiler


@six.add_metaclass(ABCMeta)
class DmModule:
    """
    Abstract class which is implemented in functional DM modules and facades.
    """
    def __init__(self, pubsub, logger):
        # type: (PubSub, Logger) -> None
        """
        :param dds: Inter-process communication interface.
        :param logger: Logging interface.
        """
        self.pubsub = pubsub
        self.logger = logger
        self.logger.info("initializing module: " + self.__class__.__name__)

    @abstractmethod
    def _start_impl(self):
        # type: () -> None
        """
        Implementation specific start script.
        """
        pass

    def start(self):
        # type: () -> None
        self.logger.info("starting module: " + self.__class__.__name__)
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
        self.logger.info("stopping module: " + self.__class__.__name__)
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
        self.logger.debug("executing periodic action at module: " + self.__class__.__name__)
        with prof.time_range(self.__class__.__name__ + ":periodic"):
            with DMProfiler(self.__class__.__name__ + ".periodic"):
                self._periodic_action_impl()
        self.logger.debug("finished periodic action at module: " + self.__class__.__name__)

