from decision_making.src.global_constants import METRIC_LOGGER_DELIMITER
from rte.python.logger.AV_logger import AV_Logger
from typing import Any, TypeVar

T = TypeVar('T', bound='MetricLogger')


class MetricLogger:

    """
    Report measurements from anywhere in the DM modules.
    Usage:
     * Obtain instance using get_logger method. Use prefix to diffrentiate between two metric loggers.
     * use 'bind' to indicate a metric to report
     * use report() to output the binded metrics

     Example:

         self._metric_logger = MetricLogger.get_logger('MY_MODULE_NAME')
         ....
         my_metric = fancy_metric()
         self._metric_logger.bind(my_metric=my_metric)
         ...
         self._metric_logger.report()
    """
    _instances = {}

    def __init__(self, prefix: str):
        self._logger = AV_Logger.get_json_logger()
        self._prefix = prefix

    @classmethod
    def get_logger(cls, prefix: str) -> T:
        """
        Obtains a MetricLogger associated with a specific prefix
        :param prefix:  is not a process-id nor an instance id. Just prefix
        :return:
        """
        if prefix not in cls._instances:
            cls._instances[prefix] = MetricLogger(prefix)
        return cls._instances[prefix]

    def report(self, message: str='', *args: Any, **kwargs: Any) -> None:
        """
        Outputs all binded data (with optional [optionally formatted] message and optional bindings)
        In most cases a periodic 'report()' call will do.
        It is also possible to add a message or additional binding
        :param message:
        :param args:  Formatting args for the message
        :param kwargs:  Any keyword=value pairs
        :return:
        """
        if len(kwargs) > 0:
            self.bind(**kwargs)
        self._logger.debug(message, *args)

    def bind(self, **kwargs: Any) -> None:
        """
        Binds the data (given as keyword, value pairs with the MetricLogger instance.
        This would be printed in the next call to 'report()'
        :param kwargs:
        :return:
        """
        self._logger.bind(**{self._prefix + METRIC_LOGGER_DELIMITER + k: v for k, v in kwargs.items()})

    def unbind(self, *args: str) -> None:
        """
        Unbinds the data (given as a list of strings) so it won't be reported on the next call to 'report()'
        :param args:
        :return:
        """
        self._logger.unbind(*[self._prefix + METRIC_LOGGER_DELIMITER + a for a in args])

