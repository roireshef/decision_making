from typing import List

from decision_making.src.manager.dm_process import DmProcess
from logging import Logger


class DmManager:
    """
    DmManager is in charge to start all dm modules and wait for them to finish
    """
    def __init__(self, logger: Logger, dm_process_list: List[DmProcess]) -> None:
        self._logger = logger
        self._dm_process_list = dm_process_list

    def start_modules(self) -> None:
        """
        start all the configured modules one by one in new processes
        :return: None
        """
        for dm_process in self._dm_process_list:
            self._logger.debug('starting DM module %s', dm_process.name)
            dm_process.start_process()

    def stop_modules(self) -> None:
        """
        signal to all the configured modules to stop their processes
        :return: 
        """
        for dm_process in self._dm_process_list:
            self._logger.debug('stopping DM module %s', dm_process.name)
            dm_process.stop_process()

        for dm_process in self._dm_process_list:
            dm_process.process.join(1)
            if dm_process.process.is_alive():
                self._logger.error('module %s has not stopped', dm_process.name)

        self._logger.debug('stopping all DM modules complete')
