import datetime
import time
from os import rename, environ

from rte.python.logger.AV_logger import AV_Logger

PROF_FILE = environ['SPAV_PATH']+'/logs/prof_log.log'
PROF_INDICATOR = 'PROF'


class DMProfiler:
    """
     Using this profiler (with DMProfiler('label'):) creates a profiling log under ultracruise/logs/prof_log.log
     Use summarize_profiler and plot_profiler to plot a certain label
    """

    _initialized = False

    def __init__(self, label):
        if DMProfiler._initialized is False:
            try:
                rename(PROF_FILE, PROF_FILE+'.old')
            except FileNotFoundError:
                pass
            with open(PROF_FILE, 'a') as f:
                f.write(f'START_TIME:{datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}\r\n')
                f.write(f'TIME_STAMP:{datetime.datetime.now().timestamp()}\r\n')
            DMProfiler._initialized = True
        self.logger = AV_Logger.get_logger()
        self.label = label.replace(' ', '_')

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, type, value, traceback):
        if traceback is not None:
            return
        current_time = time.time()
        with open(PROF_FILE, 'a+') as f:
            f.write(f'PROF:{self.label}:{current_time}:{current_time - self.start_time}\r\n')





