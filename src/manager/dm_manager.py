from decision_making.src.global_constants import *
from decision_making.src.infra.dm_factory import DmModulesEnum
from decision_making.src.manager.dm_process import DmProcess
from rte.python.logger.AV_logger import AV_Logger


class DmManager:
    _modules_list = \
        [
            DmProcess(module_type=DmModulesEnum.DM_MODULE_STATE,
                      period=STATE_MODULE_PERIOD),

            DmProcess(module_type=DmModulesEnum.DM_MODULE_BEHAVIORAL_PLANNER,
                      period=BEHAVIORAL_PLANNING_MODULE_PERIOD),

            DmProcess(module_type=DmModulesEnum.DM_MODULE_TRAJECTORY_PLANNER,
                      period=TRAJECTORY_PLANNING_MODULE_PERIOD)
        ]

    def __init__(self):
        self.logger = AV_Logger.get_logger("DM Manager")

    def start_modules(self):
        '''
        start all the configured modules one by one in new processes
        :return: None
        '''
        for dm_module in self._modules_list:
            self.logger.debug('starting DM module %s', dm_module.name)
            dm_module.start_process()

    def stop_modules(self):
        '''
        signal to all the configured modules to stop their processes
        :return: 
        '''
        for dm_module in self._modules_list:
            self.logger.debug('stopping DM module %s', dm_module.name)
            dm_module.stop_process()

        for dm_module in self._modules_list:
            dm_module.process.join(1)
            if dm_module.process.is_alive():
                self.logger.error('module %s has not stopped', dm_module.name)

        self.logger.debug('stopping all DM modules complete')


def main():

    manager = DmManager()
    manager.start_modules()

if __name__ == '__main__':
    main()

