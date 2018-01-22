import ast

import matplotlib.pyplot as plt
import numpy as np
import re

from common_data.lcm.generatedFiles.gm_lcm import LcmTrajectoryParameters
from decision_making.src.messages.dds_nontyped_message import DDSNonTypedMsg
from decision_making.src.messages.dds_typed_message import DDSTypedMsg
from decision_making.src.messages.trajectory_parameters import TrajectoryParams
from decision_making.src.state.state import State
from mapping.src.service.map_service import MapService

LOG_TIME_PATTERN = ": \d+-\d+-\d+ \d+:\d+:\d+,\d+ :"
LOG_TIME_PARSE_PATTERN = ": (\d+)-(\d+)-(\d+) (\d+):(\d+):(\d+),(\d+) :"


def parse_state_struct(str):
    state_match = re.match(pattern='[^{]*({.*})$', string=str)
    return state_match.groups()[0]

    # return timestamp in [sec]
    return (h * 3600 + m * 60 + s + 0.001 * ms) * 1.0


def parse_log_timestamp(str):
    time_match = re.match(pattern=LOG_TIME_PARSE_PATTERN, string=str)
    y, mm, d, h, m, s, ms = float(time_match.groups()[0]), float(time_match.groups()[1]), float(
        time_match.groups()[2]), float(time_match.groups()[3]), float(time_match.groups()[4]), float(
        time_match.groups()[5]), float(time_match.groups()[6])

    # return timestamp in [sec]
    return (h * 3600 + m * 60 + s + 0.001 * ms) * 1.0


def parse_state(log_content, identifier_str):
    log_timestamp = list()
    state_timestamps = list()
    states = list()

    search_pattern = ".*(%s)(.*)%s(.*)" % (LOG_TIME_PATTERN, identifier_str)
    for row in range(len(log_content)):
        state_match = re.match(pattern=search_pattern, string=log_content[row])
        if state_match is not None:
            states.append(parse_state_struct(str=state_match.groups()[2]))
            timestamp_match = re.match(pattern=".*'timestamp': (\d*)", string=state_match.groups()[2])
            timestamp = float(timestamp_match.groups()[0]) * 1E-9
            state_timestamps.append(timestamp)
            log_timestamp.append(parse_log_timestamp(state_match.groups()[0]))

    state_timestamps = np.array(state_timestamps)
    log_timestamp = np.array(log_timestamp)

    return log_timestamp, state_timestamps, states


def parse_tp(log_content):
    log_timestamp = list()
    state_timestamps = list()
    states = list()

    identifier_str = "_get_mission_params : Received mission params: "
    search_pattern = ".*(%s)(.*)%s(.*)" % (LOG_TIME_PATTERN, identifier_str)
    for row in range(len(log_content)):
        state_match = re.match(pattern=search_pattern, string=log_content[row])
        if state_match is not None:
            states.append(parse_state_struct(str=state_match.groups()[2]))
            timestamp_match = re.match(pattern=".*'time': ([0-9\.]*)", string=state_match.groups()[2])
            timestamp = float(timestamp_match.groups()[0])
            state_timestamps.append(timestamp)
            log_timestamp.append(parse_log_timestamp(state_match.groups()[0]))

    state_timestamps = np.array(state_timestamps)
    log_timestamp = np.array(log_timestamp)

    return log_timestamp, state_timestamps, states


def parse_bp(log_content):
    log_timestamp = list()
    state_timestamps = list()
    states = list()

    identifier_str = "BehavioralPlanningFacade output is "
    search_pattern = ".*(%s)(.*)%s(.*)" % (LOG_TIME_PATTERN, identifier_str)
    for row in range(len(log_content)):
        state_match = re.match(pattern=search_pattern, string=log_content[row])
        if state_match is not None:
            states.append(state_match.groups()[2])
            # timestamp_match = re.match(pattern=".*'time': ([0-9\.]*)", string=state_match.groups()[2])
            # timestamp = float(timestamp_match.groups()[0])
            timestamp = 0.0
            state_timestamps.append(timestamp)
            log_timestamp.append(parse_log_timestamp(state_match.groups()[0]))

    state_timestamps = np.array(state_timestamps)
    log_timestamp = np.array(log_timestamp)

    return log_timestamp, state_timestamps, states


def parse_no_valid_trajectories(log_content):
    log_timestamp = list()
    state_timestamps = list()
    states = list()

    identifier_str = "No valid trajectories found."
    search_pattern = ".*(%s)(.*)%s(.*)" % (LOG_TIME_PATTERN, identifier_str)
    for row in range(len(log_content)):
        state_match = re.match(pattern=search_pattern, string=log_content[row])
        if state_match is not None:
            states.append(state_match.groups()[2])
            timestamp_match = re.match(pattern=".*'ego_state': {'obj_id': \d*, 'timestamp': (\d*)",
                                       string=state_match.groups()[2])
            timestamp = float(timestamp_match.groups()[0]) * 1E-9
            state_timestamps.append(timestamp)
            log_timestamp.append(parse_log_timestamp(state_match.groups()[0]))

    state_timestamps = np.array(state_timestamps)
    log_timestamp = np.array(log_timestamp)

    return log_timestamp, state_timestamps, states


# filename = '/data/recordings/cdrive/Database/2017_12_27/logs/16_02/AV_Log_dm_main_test.log'
filename = 'C:/Users/xzjsyy/av_code/AV_Log_dm_main_test.log'

f = open(file=filename, mode='r')
log_content = f.readlines()

# State module
identifier_str = "_publish_state_if_full: publishing state "
state_module_log_timestamp, state_module_timestamps, state_module_states = parse_state(log_content=log_content,
                                                                                       identifier_str=identifier_str)

# BP module
identifier_str = "behavioral_facade.py:    76: _get_current_state  : Received State: "
bp_state_log_timestamp, bp_state_timestamps, bp_states = parse_state(log_content=log_content,
                                                                     identifier_str=identifier_str)
bp_module_log_timestamp, bp_module_timestamps, bp_module_states = parse_bp(log_content=log_content)

# TP module
identifier_str = "trajectory_planning_facade.py:   139: _get_current_state  : Received state: "
tp_state_log_timestamp, tp_state_timestamps, tp_states = parse_state(log_content=log_content,
                                                                     identifier_str=identifier_str)
tp_module_log_timestamp, tp_module_timestamps, tp_module_states = parse_tp(log_content=log_content)
no_valid_trajectories_log_timestamp, no_valid_trajectories_timestamps, no_valid_trajectories_states = parse_no_valid_trajectories(
    log_content=log_content)

# plt.plot(no_valid_trajectories_timestamps, np.ones(shape=no_valid_trajectories_timestamps.shape), '*b')
# plt.plot(state_module_timestamps[1:], np.diff(state_module_timestamps), '-.r')
# plt.plot(bp_state_timestamps[1:], np.diff(bp_state_timestamps), '-g')
# plt.plot(tp_state_timestamps[1:], np.diff(tp_state_timestamps), '-c')
# plt.plot(tp_module_timestamps[1:], np.diff(tp_module_timestamps), '-k')

tp_params_dict_1 = ast.literal_eval(tp_module_states[144])
state_dict_1 = ast.literal_eval(state_module_states[144])
tp_params_dict_2 = ast.literal_eval(tp_module_states[145])

class DdsTrajectoryParams(DDSTypedMsg, TrajectoryParams):
    pass

class DdsState(DDSNonTypedMsg, State):
    pass

tp_params_1 = DdsTrajectoryParams.deserialize(message=tp_params_dict_1)

state_1 = DdsState.deserialize(message=state_dict_1)

fig = plt.figure()
plt.plot(no_valid_trajectories_log_timestamp, np.ones(shape=no_valid_trajectories_timestamps.shape), '*b')
plt.plot(state_module_log_timestamp[1:], np.diff(state_module_timestamps), '-.r')
plt.plot(bp_state_log_timestamp[1:], np.diff(bp_state_timestamps), '-g')
plt.plot(tp_state_log_timestamp[1:], np.diff(tp_state_timestamps), '-c')
plt.plot(tp_module_log_timestamp[1:], np.diff(tp_module_timestamps), '-k')
plt.title('State time measurements')
plt.legend(('Times where no valid trajectories were found',
            'Diff between last ego timestamp in state module',
            'Diff between last ego timestamp in BP module',
            'Diff between last ego timestamp in TP module',
            'Diff between end time of high-level action'))
plt.xlabel('Log time [sec]')
plt.show()

fig = plt.figure()
plt.plot(no_valid_trajectories_log_timestamp, np.ones(shape=no_valid_trajectories_timestamps.shape), '*b')
plt.plot(state_module_log_timestamp[1:], np.diff(state_module_log_timestamp), '-.r')
plt.plot(bp_state_log_timestamp[1:], np.diff(bp_state_log_timestamp), '-g')
plt.plot(tp_state_log_timestamp[1:], np.diff(tp_state_log_timestamp), '-c')
plt.plot(tp_module_log_timestamp[1:], np.diff(tp_module_log_timestamp), '-k')
plt.plot(bp_module_log_timestamp[1:], np.diff(bp_module_log_timestamp), '-m')
plt.title('Log time measurements')
plt.legend(('Times where no valid trajectories were found',
            'Time since last state message in state module',
            'Time since last state received in BP',
            'Time since last state received in TP',
            'Time since last TP fetch of TP params',
            'Time since last output send by BP'))
plt.xlabel('Log time [sec]')
plt.show()

a = 2
