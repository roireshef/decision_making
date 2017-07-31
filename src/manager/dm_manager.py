import time
from multiprocessing import Process, Queue
from rte.python.logger.AV_logger import AV_Logger
from common_data.dds.python.Communication.ddspubsub import DdsPubSub
from decision_making.src.state.state_module import state_module


class DM_ModuleProcess():
    def __init__(self, cls, name, dds_participant, dds_file):
        self.cls = cls
        self.name = name
        self.dds_participant = dds_participant
        self.dds_file = dds_file
        self.queue = Queue()
        self.process = None

    def module_process_entry(self):
        # initialize the sub module
        logger = AV_Logger.get_logger(self.name)
        dds = DdsPubSub(self.dds_participant, self.dds_file)

        mod = self.cls(dds, logger)
        mod.start()

        # wait until a stop signal is received on the queue to stop the module
        self.queue.get()
        mod.stop()


    def start_process(self):
        self.process = Process(target = self.module_process_entry)
        self.process.start()


    def stop_process(self):
        self.queue.put(0)





class DM_Manager():
    _modules_list = \
        [
            DM_ModuleProcess(cls=state_module,
                             name='State Module',
                             dds_participant='StateParticipantLibrary::StateSubscriberParticipant',
                             dds_file='../../../common_data/dds/generatedFiles/xml/perceivedStateMain.xml')
        ]

    def __init__(self):
        self.logger = AV_Logger.get_logger("DM Manager")

    def start_modules(self):
        for dm_module in self._modules_list:
            self.logger.debug('starting DM module %s', dm_module.name)
            dm_module.start_process()

    def stop_modules(self):
        for dm_module in self._modules_list:
            self.logger.debug('stopping DM module %s', dm_module.name)
            dm_module.stop_process()

        for dm_module in self._modules_list:
            dm_module.process.join(1)
            if dm_module.process.is_alive():
                self.logger.error('module %s has not stopped', dm_module.name)

        self.logger.debug('stopping all DM modules complete')





def main():

    manager = DM_Manager()
    manager.start_modules()


    # try:
    #     while True:
    #         time.sleep(1)
    # except KeyboardInterrupt:
    #     print("stopping")
    #     manager.stop_modules()
    #     time.sleep(1)

    # time.sleep(10)
    # manager.stop_modules()
    # time.sleep(10)


if __name__ == '__main__':
    main()

