from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from decision_making.src.utils.dm_profiler import PROF_FILE, PROF_INDICATOR
from decision_making.src.utils.tabulate import tabulate


def plot_timed_series_labeled(timed_series, label):
    """
    Plot measurements according to time
    :param timed_series: an array like of tuples with (time, time_results)
    :param label: The label of the timed_series
    :return:
    """
    times, measurements = zip(*timed_series)
    plt.plot(times, measurements, label=label)


def get_profs():
    """
    :return: a dictionary with keys as labels and time_series as values
    """
    data = defaultdict(list)
    with open(PROF_FILE, 'r') as f:
        for line in f:
            if line.find(PROF_INDICATOR) == -1:
                continue
            _, label, time_stamp, instance = line.split(':')
            data[label].append((float(time_stamp), float(instance)))
    return data


def plot_profiler(label_pattern):
    """
     Plot all timed_series of a certain label_pattern
    :param label_pattern:
    :return:
    """
    profs = get_profs()
    for p, timed_series in profs.items():
        if p.find(label_pattern) != -1:
            plot_timed_series_labeled(timed_series, p)
    plt.legend()
    plt.show()


def get_start_time():
    with open(PROF_FILE, 'r') as f:
        for line in f:
            if line.find('START_TIME') != -1:
                return ':'.join(line.split(':')[1:])


def summarize_profiler():
    profs = get_profs()
    print(f' Profiling data date: {get_start_time()}')
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
    plot_profiler('')
