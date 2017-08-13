from decision_making.src.global_constants import *
from decision_making.src.infra.dm_factory import DmModulesEnum
from decision_making.src.manager.dm_process import DmProcess
from decision_making.src.manager.dm_trigger import DmTriggerType
from rte.python.logger.AV_logger import AV_Logger


class DmManager:

    def __init__(self):
        self.logger = AV_Logger.get_logger(DM_MANAGER_NAME_FOR_LOGGING)
        self.modules_list = \
            [
                DmProcess(module_type=DmModulesEnum.DM_MODULE_STATE,
                          trigger_type=DmTriggerType.DM_TRIGGER_NONE,
                          trigger_args={}),

                DmProcess(module_type=DmModulesEnum.DM_MODULE_BEHAVIORAL_PLANNER,
                          trigger_type=DmTriggerType.DM_TRIGGER_PERIODIC,
                          trigger_args={'period':BEHAVIORAL_PLANNING_MODULE_PERIOD}),

                DmProcess(module_type=DmModulesEnum.DM_MODULE_TRAJECTORY_PLANNER,
                          trigger_type=DmTriggerType.DM_TRIGGER_PERIODIC,
                          trigger_args={'period':TRAJECTORY_PLANNING_MODULE_PERIOD})
            ]

    def start_modules(self):
        """
        start all the configured modules one by one in new processes
        :return: None
        """
        for dm_module in self.modules_list:
            self.logger.debug('starting DM module %s', dm_module.get_name())
            dm_module.start_process()

    def stop_modules(self):
        """
        signal to all the configured modules to stop their processes
        :return: 
        """
        for dm_module in self.modules_list:
            self.logger.debug('stopping DM module %s', dm_module.get_name())
            dm_module.stop_process()

        for dm_module in self.modules_list:
            dm_module.process.join(1)
            if dm_module.process.is_alive():
                self.logger.error('module %s has not stopped', dm_module.get_name())

        self.logger.debug('stopping all DM modules complete')

