import sys
import numpy as np
import matplotlib.pyplot as plt

from decision_making.src.messages.trajectory_plan_message import TrajectoryPlanMsg
from decision_making.src.planning.types import C_V
from decision_making.src.state.state import State
from decision_making.test.log_analysis.log_messages import LogMsg
from decision_making.test.log_analysis.parse_log_messages import DmLogParser, \
    STATE_IDENTIFIER_STRING_BP, STATE_IDENTIFIER_STRING_TP, STATE_IDENTIFIER_STRING_STATE_MODULE, \
    STATE_IDENTIFIER_STRING_STATE_MODULE_SIMULATION

LOG_PATH_FOR_ANALYSIS = '/home/vzt9fl/av_code/spav/logs/AV_Log_rbcar0_sim_mimicking_2018_03_19_11_18_start_52_one_lane.log'
LOG_PATH_FOR_ANALYSIS2 = '/home/vzt9fl/av_code/spav/logs/AV_Log_rbcar0_2018_03_19_11_18_start_52_one_lane.log'

def main():
    filename = LOG_PATH_FOR_ANALYSIS if ('log_filename' not in sys.argv) else sys.argv['log_filename']
    f = open(file=filename, mode='r')
    log_content = f.readlines()

    filename2 = LOG_PATH_FOR_ANALYSIS2 if ('log_filename2' not in sys.argv) else sys.argv['log_filename2']
    f2 = open(file=filename2, mode='r')
    log_content2 = f2.readlines()


    ###########################
    # Load relevant messages
    ###########################

    # State module
    state_module_log_timestamp, state_module_timestamps, state_module_states = DmLogParser.parse_state_message(
        log_content=log_content,
        identifier_str=STATE_IDENTIFIER_STRING_STATE_MODULE_SIMULATION)


    state_module_log_timestamp2, state_module_timestamps2, state_module_states2 = DmLogParser.parse_state_message(
        log_content=log_content2,
        identifier_str=STATE_IDENTIFIER_STRING_STATE_MODULE_SIMULATION)

    ###########################
    # Plot log analysis
    ###########################


    fig = plt.subplot(211)

    # Plot ego velocity
    actual_v = []
    ego_timestamps = []
    for state_message_index in range(len(state_module_states)):
        # Convert log messages to dict
        state_msg = LogMsg.convert_message_to_dict(state_module_states[state_message_index])
        # Deserialize from dict to object
        state = LogMsg.deserialize(class_type=State, message=state_msg)  # type: State
        if state_message_index == 0:
            baseline_timestamp = state.ego_state.timestamp_in_sec
        actual_v.append(state.ego_state.v_x)
        ego_timestamps.append(state.ego_state.timestamp_in_sec - baseline_timestamp)

    plt.plot(ego_timestamps, actual_v, '-b')

    actual_v = []
    ego_timestamps = []
    for state_message_index in range(len(state_module_states2)):
        # Convert log messages to dict
        state_msg = LogMsg.convert_message_to_dict(state_module_states2[state_message_index])
        # Deserialize from dict to object
        state = LogMsg.deserialize(class_type=State, message=state_msg)  # type: State
        if state_message_index == 0:
            baseline_timestamp = state.ego_state.timestamp_in_sec
        actual_v.append(state.ego_state.v_x)
        ego_timestamps.append(state.ego_state.timestamp_in_sec - baseline_timestamp)

    plt.plot(ego_timestamps, actual_v, '-r')
    plt.ylabel('$m \cdot s^{-1}$')



    plt.subplot(212)

    # Plot ego acceleration
    accel = []
    ego_timestamps = []
    for state_message_index in range(len(state_module_states)):
        # Convert log messages to dict
        state_msg = LogMsg.convert_message_to_dict(state_module_states[state_message_index])
        # Deserialize from dict to object
        state = LogMsg.deserialize(class_type=State, message=state_msg)  # type: State
        if state_message_index == 0:
            baseline_timestamp = state.ego_state.timestamp_in_sec
        accel.append(state.ego_state.acceleration_lon)
        ego_timestamps.append(state.ego_state.timestamp_in_sec - baseline_timestamp)

    plt.plot(ego_timestamps, accel, '-b')

    # Plot ego acceleration
    accel = []
    ego_timestamps = []
    for state_message_index in range(len(state_module_states2)):
        # Convert log messages to dict
        state_msg = LogMsg.convert_message_to_dict(state_module_states2[state_message_index])
        # Deserialize from dict to object
        state = LogMsg.deserialize(class_type=State, message=state_msg)  # type: State
        if state_message_index == 0:
            baseline_timestamp = state.ego_state.timestamp_in_sec
        accel.append(state.ego_state.acceleration_lon)
        ego_timestamps.append(state.ego_state.timestamp_in_sec - baseline_timestamp)

    plt.plot(ego_timestamps, accel, '-r')
    plt.ylabel('$m \cdot s^{-2}$')
    plt.xlabel('seconds')

    plt.show()

if __name__ == '__main__':
    main()
