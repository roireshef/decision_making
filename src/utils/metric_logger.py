from rte.python.logger.AV_logger import AV_Logger


class MetricLogger:
    """
    Report measurements from anywhere in the DM modules.
    """
    _instances = {}

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
        self._logger.bind(**{self._prefix + '_' + k:v for k, v in kwargs.items()})

    def unbind(self, *args):
        self._logger.unbind(*[self._prefix + '_' + a for a in args])

