import matplotlib.pyplot as plt
import numpy as np

from decision_making.paths import Paths
from decision_making.src.planning.behavioral.default_config import DEFAULT_ACTION_SPEC_FILTERING


def plot_filters_map(log_file_path: str):
    """
    Plot various graphs concerning localization, BP and TP outputs
    :param log_file_path: path to AV_Log_dm_main.log file
    :return: a showable matplotlib figure
    """
    file = open(log_file_path, 'r')
    f = plt.figure(1)
    plt.rcParams['image.cmap'] = 'jet'
    colors_num = len(DEFAULT_ACTION_SPEC_FILTERING._filters) + 1

    for idx, filter in enumerate(DEFAULT_ACTION_SPEC_FILTERING._filters + ['Passed']):
        r = idx/colors_num
        color = plt.cm.hsv(r)
        plt.scatter(x=[9.5], y=[idx*8], c=np.array([color]), linewidths=0)
        plt.text(9.6, idx*8, filter.__str__())

    while True:
        text = file.readline()
        if not text:
            break

        if 'Filtering_map' in text:
            colon_str = text.split('timestamp_in_sec ')[1].split(':')
            timestamp = float(colon_str[0])
            filters_result = list(map(int, colon_str[1].replace('array([', '').replace('])', '').split(', ')))
            # filtering_map.append((timestamp, filters_result))
            colors = plt.cm.hsv(np.array(filters_result)/colors_num)
            plt.scatter(np.full(len(filters_result), timestamp), np.array(range(len(filters_result))),
                        c=colors, linewidths=0)

    plt.xlabel('time [s]')
    plt.ylabel('action')
    return f


if __name__ == "__main__":
    # Enter path of log file to analyze here:
    file_path = '%s/../logs/AV_Log_dm_main.log' % Paths.get_repo_path()
    f = plot_filters_map(file_path)
    plt.show(f)
