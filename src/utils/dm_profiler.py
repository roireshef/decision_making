from rte.python.logger.AV_logger import AV_Logger
import time


class DMProfiler:

    def __init__(self, label):
        self.logger = AV_Logger.get_logger()
        self.label = label.replace(' ','_')

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, type, value, traceback):
        self.logger.error(f'PROF_{self.label}:{time.time() - self.start_time}')



if __name__ == '__main__':
    with DMProfiler('lab1'):
        time.sleep(1)

    with DMProfiler('lab2'):
        time.sleep(2)

