import time
from multiprocessing import Process, Event
from typing import Callable
from decision_making.src.infra.dm_module import DmModule
from decision_making.src.manager.dm_trigger import DmTriggerType, DmPeriodicTimerTrigger, DmNullTrigger
from rte.python.os import catch_interrupt_signals
from rte.python.logger.AV_logger import AV_Logger
from os import getpid
import rte.python.profiler as prof


class DmProcess:

    def __init__(self, module_creation_method: Callable[[], DmModule], trigger_type: DmTriggerType, trigger_args: dict,
                 name: str):
        """
        Manager for a single DM module running in a separate process
        :param module_creation_method: the method to create an instance of the DM module to run in a separate process
        :param trigger_type: the type of trigger to use
        :param trigger_args: dictionary containing keyword arguments for initializing the trigger
        """

        self._module_creation_method = module_creation_method
        self._trigger_type = trigger_type
        self._trigger_args = trigger_args
        self.name = name
        self._stop_event = Event()

        self._process_name = "DM_process_{}".format(self.name)
        self.process = Process(target=self._module_process_entry, name=self._process_name, daemon=True)

        self._trigger = None
        self._module_instance = None

        self.logger = AV_Logger.get_logger(self._process_name)

    def start_process(self) -> None:
        """
        Create and start a process for the DM module
        :return:
        """
        self.process.start()

    def stop_process(self) -> None:
        """
        API for signaling to the DM module's process to stop
        :return: None
        """
        self._stop_event.set()

    def _module_process_entry(self) -> None:
        """
        Entry method to the process created for the DM module.
        This is the first code the new process will call, it starts the module, activates the trigger, and waits
        until it receives a stop signal.
        :return: None
        """

        # store process id (pid) used for tracking process activity in log
        self._pid = getpid()
        self.logger.debug('%d: started "%s"', self._pid, self.name)

        # stop process if an interrupt OS signal is received
        catch_interrupt_signals()

        # init profiling if enabled
        prof.start()

        # create the trigger and activate it.
        if self._trigger_type == DmTriggerType.DM_TRIGGER_PERIODIC:
            self._trigger = DmPeriodicTimerTrigger(self._trigger_callback, self._process_name, **self._trigger_args)
        elif self._trigger_type == DmTriggerType.DM_TRIGGER_NONE:
            self._trigger = DmNullTrigger()

        self._module_instance = self._module_creation_method()

        # create the sub module
        self._module_instance.start()

        # activate method can be blocking, depending on the trigger type
        self._trigger.activate()

        # wait until a stop signal is received on the queue to stop the module
        self._module_process_wait_for_signal()

    def _module_process_wait_for_signal(self) -> None:
        try:
            self._stop_event.wait()
        except KeyboardInterrupt:
            self.logger.debug('%d: stop signal received', self._pid)
            # after a stop signal was received we should perform the exit flow
            self._module_process_exit()

    def _module_process_exit(self) -> None:
        """
        perform the actions necessary to stop the DM module running inside the process
        :return: None
        """
        if self._trigger.is_active():
            self._trigger.deactivate()
        self._module_instance.stop()
        prof.cleanup()

    def _trigger_callback(self) -> None:
        """
        __timer_callback - this method runs in the module's process
        :return: None
        """
        try:
            self._module_instance.periodic_action()
        except KeyboardInterrupt:
            self.logger.debug('%d: stop signal received', self._pid)
            # after a stop signal was received we should perform the exit flow
            self._module_process_exit()
