from decision_making.src import global_constants
from decision_making.src.dm_main import DmInitialization
from decision_making.src.global_constants import BEHAVIORAL_PLANNING_MODULE_PERIOD, TRAJECTORY_PLANNING_MODULE_PERIOD, \
    DM_MANAGER_NAME_FOR_LOGGING
from decision_making.src.manager.dm_manager import DmManager
from decision_making.src.manager.dm_process import DmProcess
from decision_making.src.manager.dm_trigger import DmTriggerType
from decision_making.test import constants
from os import getpid
from rte.python.logger.AV_logger import AV_Logger
from rte.python.os import catch_interrupt_signals


def main():
    """
    initializes DM planning pipeline. for switching between BP/TP impl./mock make sure to comment out the relevant
    instantiation in modules_list.
    """
    modules_list = \
        [
            DmProcess(DmInitialization.create_behavioral_planner,
                      trigger_type=DmTriggerType.DM_TRIGGER_PERIODIC,
                      trigger_args={'period': BEHAVIORAL_PLANNING_MODULE_PERIOD},
                      name="name1"),

            DmProcess(DmInitialization.create_trajectory_planner,
                      trigger_type=DmTriggerType.DM_TRIGGER_PERIODIC,
                      trigger_args={'period': TRAJECTORY_PLANNING_MODULE_PERIOD},
                      name="name2")
        ]
    logger = AV_Logger.get_logger(DM_MANAGER_NAME_FOR_LOGGING)
    logger.debug('%d: (main) registered signal handler', getpid())
    logger.info("DM Global Constants: %s", {k: v for k, v in global_constants.__dict__.items() if k.isupper()})
    logger.info("DM Test Constants: %s", {k: v for k, v in constants.__dict__.items() if k.isupper()})

    catch_interrupt_signals()
    manager = DmManager(logger, modules_list)
    manager.start_modules()
    try:
        manager.wait_for_submodules()
    except KeyboardInterrupt:
        logger.info('%d: (main) Interrupted by signal!', getpid())
        pass
    finally:
        manager.stop_modules()


if __name__ == '__main__':
    main()
