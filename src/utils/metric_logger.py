from rte.python.logger.AV_logger import AV_Logger


class MetricLogger:
    """
    Report measurements from anywhere in the DM modules.
    """
    _instance = None

    def __init__(self):
        self._logger = AV_Logger.get_json_logger()

    @classmethod
    def get_logger(cls):
        if cls._instance is None:
            cls._instance = MetricLogger()
        return cls._instance

    def report(self, message='', *args, **kwargs):
        if len(kwargs) > 0:
            self.bind(**kwargs)
        self._logger.debug(message, *args)

    def bind(self, **kwargs):
        self._logger.bind(**kwargs)

    def unbind(self, *args):
        self._logger.unbind(*args)

