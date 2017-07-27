from ddspubsub import DdsPubSub
from rte.python.logger.AV_logger import AV_Logger

class DM_Module():

    def __init__(self, DDS : DdsPubSub, logger):
        self.DDS = DDS
        self.logger = logger

    def start(self):
        self.logger.info('starting')

    def stop(self):
        self.logger.info('stopping')

    def tick(self):
        self.logger.info('tick')
