from multiprocessing import Queue, Process

from decision_making.src.infra.dm_factory import DmModulesEnum, DmModuleFactory
from rte.python.periodic_timer.periodic_timer import PeriodicTimer


class DmProcess():
    def __init__(self, module_type : DmModulesEnum, period: float) -> None:
        '''
        Manager for a single DM module running in a separate process
        :param cls: the class of the DM module to be instantiated
        :param name: a string name of the module
        :param period: time period in seconds for calling the modules periodic_action method. 
                    if period=0 then periodic_action will not be called.
        :param dds_participant: the name of the DDS participant
        :param dds_file: the DDS XML generated file name
        '''
        self.module_type = module_type
        self.period = period
        self.queue = Queue()
        self.process = None
        self.module_instance = None

    def start_process(self):
        '''
        Create and start a process for the DM module
        :return: 
        '''
        process_name = "DM_process_{}".format(self.module_type)
        self.process = Process(target=self.__module_process_entry, name=process_name)
        self.process.start()

    def stop_process(self):
        '''
        signal to the DM mmodule's process to stop 
        :return: None
        '''
        self.queue.put(0)

    def __module_process_entry(self):
        '''
        Entry method to the process created for the DM module.
        The module initialization should be done inside the new process.
        :return: None
        '''
        # create the sub module
        self.module_instance = DmModuleFactory.create_dm_module(self.module_type)
        self.module_instance.start()

        # create a timer for the module only if the period is defined
        if self.period > 0:
            self.timer = PeriodicTimer(self.period, self.__timer_callback)
            self.timer.start(run_in_thread=False)

        # wait until a stop signal is received on the queue to stop the module
        self.queue.get()
        self.module_instance.stop()

    def __timer_callback(self):
        '''
        __timer_callback - this method runs in the module's process
        :return: None
        '''
        self.module_instance.periodic_action()
        # check if a stop signal was received
        if not self.queue.empty():
            self.module_instance.stop()
