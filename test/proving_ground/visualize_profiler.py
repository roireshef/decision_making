import ast
from collections import defaultdict
import re

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
    flag_strs = {
        "Searcher is idle, preparing new search": -0.1,
        "Longitudinal driver override": 0.6,
        "Got invalid driving plan": 0.4,
    }
    flags = {}
    for name in flag_strs:
        flags[name] = False
    vals_ptr = [
        ("search_iteration_duration", re.compile("Search iteration .+ finished, duration: (.*)")),
    ]
    vals = {}
    for name, _ in vals_ptr:
        vals[name] = 0.0

    prev_t = 0.0
    dur_sum = 0.0
    with open(filename, 'r') as f:
        for line in f:
            prefix_index = line.find(LOG_MSG_PROFILER_PREFIX)
            if prefix_index > -1:
                msg_dict = ast.literal_eval(line[prefix_index+len(LOG_MSG_PROFILER_PREFIX):])
                data[msg_dict['label']].append((float(msg_dict['current_time']), float(msg_dict['running_time'])))
                if msg_dict['label'] == "SubdivisionModule.periodic":
                    if prev_t != 0.0:
                        for name, value in flags.items():
                            data[name].append((prev_t, value * flag_strs[name]))
                            flags[name] = False
                        # for name, value in vals.items():
                        #     data[name].append((prev_t, value))

                    prev_t = float(msg_dict['current_time'])
                    data["sum_iteration_duration"].append((prev_t, dur_sum))
                    dur_sum = 0.0
            for name in flag_strs:
                if name in line:
                    flags[name] = True
            for name, ptr in vals_ptr:
                ma = ptr.findall(line)
                if ma:
                    vals[name] = float(ma[0])
                    if name == "search_iteration_duration":
                        dur_sum += vals[name]
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
        # if p.find(label_pattern) != -1:
        if p in label_pattern:
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
    wanted_labels = [
        'MapProvider.recv',
         'MapProvider.periodic',
         'MapProvider.deserialize',
         'SceneStaticMapGenerator.update_map_aux',
         'MapProvider._route_planner.plan',
         'MapProvider.feed_map_queue',
         'MapProvider.feed_route_plan_queue',
         'SubdivisionModule.recorder_flush',
         'SubdivisionModule._get_current_route_plan',
         'SubdivisionModule.get_latest_sample',
         'SubdivisionModule.deserialize',
         # 'SubdivisionModule._get_current_pedal_position',
         # 'SubdivisionModule._get_pedal_pressed_status',
         # 'SubdivisionModule._get_current_torque_request_status',
         # 'SubdivisionModule._get_current_control_status',
         # 'SubdivisionModule._get_current_map',
         # 'SubdivisionModule._get_current_turn_signal',
         # 'SubdivisionModule._get_current_gap_setting',
         # 'SubdivisionModule._get_current_set_speed',
         # '_get_current_dynamic_traffic_control_device_status.get_latest_sample',
         # '_get_current_dynamic_traffic_control_device_status.deserialize',
         # 'SubdivisionModule._get_current_dynamic_traffic_control_device_status',
         # 'SubdivisionModule._get_current_scene_dynamic',
         # 'SubdivisionModule._get_lane_change_for_time_and_route',
         'SubdivisionModule.reload_config',
         'SubdivisionModule.convert_state',
         'SubdivisionModule.publish_state',
         'SubdivisionModule.run_search',
         'SubdivisionModule.plan',
         'SubdivisionModule.publish_action',
         'SubdivisionModule.convert_action',
         'SubdivisionModule.publish_outputs',
         'SubdivisionModule.periodic',
         'Searcher is idle,'
         'preparing new search',
         'Longitudinal driver override',
         'Got invalid driving plan',
         'search_iteration_duration',
         'sum_iteration_duration',
         'SubdivisionModule.exception',
    ]

    plot_profiler(profs, wanted_labels)
