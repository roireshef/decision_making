from rte.python.logger.AV_logger import AV_Logger


class MetricLogger:
    """
    Report measurements from anywhere in the DM modules.
    """
    _instance = None

    def __init__(self,component_name):
        self._logger = AV_Logger.get_json_logger()
        self._component_name = component_name

    @classmethod
    def get_logger(cls,component_name):
        if cls._instance is None:
            cls._instance = MetricLogger(component_name)
        return cls._instance

    def report(self, message='', *args, **kwargs):
        if len(kwargs) > 0:
            self.bind(**kwargs)
        self._logger.debug(message, *args)

    def bind(self, **kwargs):
        self._logger.bind(**{self._component_name+'_'+k:v for k,v in kwargs.items()})

    def unbind(self, *args):
        self._logger.unbind(*[self._component_name+'_'+a for a in args])

