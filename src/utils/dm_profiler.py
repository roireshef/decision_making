from collections import defaultdict
import matplotlib.pyplot as plt
from rte.python.logger.AV_logger import AV_Logger
import time
import math
import numpy as np
from os import rename, environ
from decision_making.src.utils.tabulate import tabulate

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



def plot_timed_series_labeled(timed_series, label, **kwargs):
    times, measurements = zip(*timed_series)
    plt.plot(times, measurements, label=label, **kwargs)


def get_profs():
    data = defaultdict(list)
    with open(PROF_FILE, 'r') as f:
        for line in f:
            if line.find(PROF_INDICATOR) == -1:
                continue
            _, label, time_stamp, instance = line.split(':')
            data[label].append((time_stamp, float(instance)))
    return data

def plot_profiler():
    profs = get_profs()
    for p, timed_series in profs.items():
        plot_timed_series_labeled(timed_series, p)
    plt.legend()

def summarize_profiler():
    profs = get_profs()
    data=[]
    headers = ['label', '#calls' , 'avg.time', 'max.time', 'stdev.', 'cumulative_time']
    for p, timed_series in profs.items():
        cumulative_time = sum([t for _, t in timed_series])
        max_time = max([t for _, t in timed_series])
        std = np.std([t for _, t in timed_series])
        data.append([p,len(timed_series), cumulative_time/len(timed_series), max_time, std ,cumulative_time])

    print(tabulate(sorted(data, key=lambda x:x[-1], reverse=True), headers))


if __name__ == '__main__':
    summarize_profiler()
#    with DMProfiler('lab1'):
#        time.sleep(1)
#
#    with DMProfiler('lab2'):
#        time.sleep(2)

