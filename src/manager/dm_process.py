from multiprocessing import Queue, Process

from decision_making.src.infra.dm_factory import DmModulesEnum, DmModuleFactory
from decision_making.src.manager.dm_trigger import DmTriggerType, DmPeriodicTimerTrigger

class DmProcess():
    def __init__(self, module_type: DmModulesEnum, trigger_type: DmTriggerType, trigger_args: dict) -> None:
        """
        Manager for a single DM module running in a separate process
        :param module_type: the type of the DM module to be instantiated
        :param trigger_type: the type of trigger to use
        :param trigger_args: dictionary containing keyword arguments for initializing the trigger
        """
        self.module_type = module_type
        self.trigger_type = trigger_type
        self.trigger_args = trigger_args
        self.queue = Queue()
        self.process = None
        self.module_instance = None
        self.trigger = None

    def get_name(self):
        return str(self.module_type)

    def start_process(self):
        """
        Create and start a process for the DM module
        :return: 
        """
        process_name = "DM_process_{}".format(self.module_type)
        self.process = Process(target=self.__module_process_entry, name=process_name)
        self.process.start()

    def stop_process(self):
        """
        signal to the DM module's process to stop
        :return: None
        """
        self.queue.put(0)

    def __module_process_entry(self):
        """
        Entry method to the process created for the DM module.
        The module initialization should be done inside the new process.
        :return: None
        """
        # create the sub module
        self.module_instance = DmModuleFactory.create_dm_module(self.module_type)
        self.module_instance.start()

        # create the trigger and activate it.
        # It is important to create the trigger inside the new process!!
        if self.trigger_type == DmTriggerType.DM_TRIGGER_PERIODIC:
            self.trigger = DmPeriodicTimerTrigger(self.__trigger_callback, **self.trigger_args)

        # activate method can be blocking, depending on the trigger type
        self.trigger.activate()

        # wait until a stop signal is received on the queue to stop the module
        self.queue.get()
        if self.trigger.is_active():
            self.trigger.deactivate()
        self.module_instance.stop()

    def __trigger_callback(self):
        """
        __timer_callback - this method runs in the module's process
        :return: None
        """
        self.module_instance.periodic_action()
        # check if a stop signal was received (necessary in case the trigger method is blocking)
        if not self.queue.empty():
            if self.trigger.is_active():
                self.trigger.deactivate()
            self.module_instance.stop()
