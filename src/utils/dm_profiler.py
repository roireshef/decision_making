import time

from decision_making.src.global_constants import LOG_MSG_PROFILER_PREFIX
from rte.python.logger.AV_logger import AV_Logger


class DMProfiler:
    """
     Using this profiler (with DMProfiler('label'):) creates a profiling log under ultracruise/logs/prof_log.log
     Use summarize_profiler and plot_profiler to plot a certain label
    """
    logger = AV_Logger.get_logger()

    def __init__(self, label):
        self.logger = AV_Logger.get_logger()
        self.label = label.replace(' ', '_')

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, type, value, traceback):
        current_time = time.time()
        self.logger.debug("%s{'current_time': %s, 'label': '%s', 'running_time': %s}" %
                          (LOG_MSG_PROFILER_PREFIX, current_time, self.label, current_time - self.start_time))

    @staticmethod
    def profile(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            current_time = time.time()
            DMProfiler.logger.debug("%s{'current_time': %s, 'label': '%s', 'running_time': %s}" %
                              (LOG_MSG_PROFILER_PREFIX, current_time, str(func), current_time - start_time))
            return result
        return wrapper
