import matplotlib.pyplot as plt
import numpy as np
import re


def parse_state(log_content):
    state_timestamps = list()
    states = list()

    identifier_str = "_publish_state_if_full: publishing state "
    search_pattern = "(.*)%s(.*)" % identifier_str
    for row in range(len(log_content)):
        state_match = re.match(pattern=search_pattern, string=log_content[row])
        if state_match is not None:
            states.append(state_match.groups()[1])
            timestamp_match = re.match(pattern=".*'timestamp': (\d*)", string=state_match.string)
            timestamp = float(timestamp_match.groups()[0])*1E-9
            state_timestamps.append(timestamp)

    state_timestamps = np.array(state_timestamps)

    return state_timestamps, states

def parse_tp(log_content):
    state_timestamps = list()
    states = list()

    identifier_str = "_get_mission_params : Received mission params: "
    search_pattern = "(.*)%s(.*)" % identifier_str
    for row in range(len(log_content)):
        state_match = re.match(pattern=search_pattern, string=log_content[row])
        if state_match is not None:
            states.append(state_match.groups()[1])
            timestamp_match = re.match(pattern=".*'time': ([0-9\.]*)", string=state_match.string)
            timestamp = float(timestamp_match.groups()[0])
            state_timestamps.append(timestamp)

    state_timestamps = np.array(state_timestamps)

    return state_timestamps, states


def parse_no_valid_trajectories(log_content):
    state_timestamps = list()
    states = list()

    identifier_str = "No valid trajectories found."
    search_pattern = "(.*)%s(.*)" % identifier_str
    for row in range(len(log_content)):
        state_match = re.match(pattern=search_pattern, string=log_content[row])
        if state_match is not None:
            states.append(state_match.groups()[1])
            timestamp_match = re.match(pattern=".*'ego_state': {'obj_id': \d*, 'timestamp': (\d*)", string=state_match.string)
            timestamp = float(timestamp_match.groups()[0])*1E-9
            state_timestamps.append(timestamp)

    state_timestamps = np.array(state_timestamps)

    return state_timestamps, states

filename = '/data/recordings/cdrive/Database/2017_12_27/logs/16_02/AV_Log_dm_main_test.log'

f = open(file=filename, mode='r')
log_content = f.readlines()

state_module_timestamps, state_module_states = parse_state(log_content=log_content)
tp_module_timestamps, tp_module_states = parse_tp(log_content=log_content)
no_valid_trajectories_timestamps, no_valid_trajectories_states = parse_no_valid_trajectories(log_content=log_content)

plt.plot(no_valid_trajectories_timestamps, np.ones(shape=no_valid_trajectories_timestamps.shape), '*b')
plt.plot(state_module_timestamps[1:], np.diff(state_module_timestamps), '-r')
plt.plot(tp_module_timestamps[1:], np.diff(tp_module_timestamps), '-k')
plt.draw()




