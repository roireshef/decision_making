import ast
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from decision_making.paths import Paths
from decision_making.src.global_constants import LOG_MSG_PROFILER_PREFIX
from tabulate import tabulate


def plot_timed_series_labeled(timed_series, label):
    """
    Plot measurements according to time
    :param timed_series: an array like of tuples with (time, time_results)
    :param label: The label of the timed_series
    :return:
    """
    times, measurements = zip(*timed_series)
    return plt.plot(times, measurements, label=label)


def get_profs(filename):
    """
    :return: a dictionary with keys as labels and time_series as values
    """
    data = defaultdict(list)
    with open(filename, 'r') as f:
        for line in f:
            prefix_index = line.find(LOG_MSG_PROFILER_PREFIX)
            if prefix_index > -1:
                msg_dict = ast.literal_eval(line[prefix_index+len(LOG_MSG_PROFILER_PREFIX):])
                data[msg_dict['label']].append((float(msg_dict['current_time']), float(msg_dict['running_time'])))
    return data


def plot_profiler(profs, label_pattern):
    """
     Plot all timed_series of a certain label_pattern
    :param label_pattern:
    :return:
    """
    fig = plt.figure()

    lines = []
    for p, timed_series in profs.items():
        if p.find(label_pattern) != -1:
            line = plot_timed_series_labeled(timed_series, p)
            lines.append(line[0])
    leg = plt.legend()

    lined = dict()
    for legline, origline in zip(leg.get_lines(), lines):
        legline.set_picker(5)  # 5 pts tolerance
        lined[legline] = origline

    def onpick(event):
        # on the pick event, find the orig line corresponding to the
        # legend proxy line, and toggle the visibility
        legline = event.artist
        origline = lined[legline]
        vis = not origline.get_visible()
        origline.set_visible(vis)
        # Change the alpha on the line in the legend so we can see what lines
        # have been toggled
        if vis:
            legline.set_alpha(1.0)
        else:
            legline.set_alpha(0.2)
        fig.canvas.draw()

    fig.canvas.mpl_connect('pick_event', onpick)

    plt.show()

def summarize_profiler(profs):
    """
    Outputs statistics summary table for all profiled code-snippets
    """
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
    file_path = '%s/../logs/AV_Log_lba_main.log' % Paths.get_repo_path()

    # read data
    profs = get_profs(file_path)

    # output table of statistics
    summarize_profiler(profs)

    # A search string for the label can be provided
    plot_profiler(profs, '')
