from rte.python.logger.AV_logger import AV_Logger
from ddspubsub import DdsPubSub
from src.state.state_module import state_module
import time


def main():
    logger = AV_Logger.get_logger("DM Manager")
    state_dds = DdsPubSub("StateParticipantLibrary::StateSubscriberParticipant",
                    '../../../common_data/dds/generatedFiles/xml/perceivedStateMain.xml')

    state = state_module(state_dds, logger)

    state.start()

    time.sleep(20)

    state.stop()

    # logger.info("hello from DM Manager")



if __name__ == '__main__':
    main()

