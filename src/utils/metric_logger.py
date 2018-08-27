from rte.python.logger.AV_logger import AV_Logger


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
    _DELIM = '_'

    def __init__(self,prefix):
        self._logger = AV_Logger.get_json_logger()
        self._prefix = prefix

    @classmethod
    def get_logger(cls,prefix):
        """
        :param prefix:  is not a process-id nor an instance id. Just prefix
        :return:
        """
        if prefix not in cls._instances:
            cls._instances[prefix] = MetricLogger(prefix)
        return cls._instances[prefix]

    def report(self, message='', *args, **kwargs):
        if len(kwargs) > 0:
            self.bind(**kwargs)
        self._logger.debug(message, *args)

    def bind(self, **kwargs):
        self._logger.bind(**{self._prefix + self._DELIM + k:v for k, v in kwargs.items()})

    def unbind(self, *args):
        self._logger.unbind(*[self._prefix + self._DELIM + a for a in args])

