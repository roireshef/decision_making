import time
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from decision_making.src.utils.tabulate import tabulate
from os import rename, environ
from rte.python.logger.AV_logger import AV_Logger

PROF_FILE = environ['SPAV_PATH']+'/logs/prof_log.log'
PROF_INDICATOR = 'PROF'


class DMProfiler:

    _initialized = False

    def __init__(self, label):
        if DMProfiler._initialized is False:
            try:
                rename(PROF_FILE, PROF_FILE+'.old')
            except FileNotFoundError:
                pass
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



def plot_timed_series_labeled(timed_series, label):
    times, measurements = zip(*timed_series)
    plt.plot(times, measurements, label=label)


def get_profs():
    data = defaultdict(list)
    with open(PROF_FILE, 'r') as f:
        for line in f:
            if line.find(PROF_INDICATOR) == -1:
                continue
            _, label, time_stamp, instance = line.split(':')
            data[label].append((float(time_stamp), float(instance)))
    return data

def plot_profiler(label_pattern):
    profs = get_profs()
    for p, timed_series in profs.items():
        if p.find(label_pattern) != -1:
            plot_timed_series_labeled(timed_series, p)
    plt.legend()
    plt.show()


def summarize_profiler():
    profs = get_profs()
    data = []
    headers = ['label', '#calls', 'avg.time', 'max.time', 'stdev.', '25%', '50%', '75%', '95%', 'cumulative_time']
    for p, timed_series in profs.items():
        time_instances = [t for _, t in timed_series]
        cumulative_time = sum(time_instances)
        max_time = max(time_instances)
        std = np.std(time_instances)
        data.append([p, len(timed_series), cumulative_time/len(timed_series), max_time, std ,
                     np.percentile(time_instances, 25),
                     np.percentile(time_instances, 50),
                     np.percentile(time_instances, 75),
                     np.percentile(time_instances, 95), cumulative_time])

    print(tabulate(sorted(data, key=lambda x: x[-1], reverse=True), headers))


if __name__ == '__main__':
    summarize_profiler()
    plot_profiler('get_scene_static')



