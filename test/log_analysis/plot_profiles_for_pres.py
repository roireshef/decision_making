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

#LOG_PATH_FOR_ANALYSIS = '/local/flash/recordings/2018_03_26/2018_03_26_16_28/AV_Log_dm_main.log'
LOG_PATH_FOR_ANALYSIS = '/home/nz2v30/av_code/spav/logs/AV_Log_rbcar0.log'

def rng(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def main():
    filename = LOG_PATH_FOR_ANALYSIS if ('log_filename' not in sys.argv) else sys.argv['log_filename']
    f = open(file=filename, mode='r')
    log_content = f.readlines()

    # filename2 = LOG_PATH_FOR_ANALYSIS2 if ('log_filename2' not in sys.argv) else sys.argv['log_filename2']
    # f2 = open(file=filename2, mode='r')
    # log_content2 = f2.readlines()


    ###########################
    # Load relevant messages
    ###########################

    # State module
    state_module_log_timestamp, state_module_timestamps, state_module_states = DmLogParser.parse_state_message(
        log_content=log_content,
        identifier_str=STATE_IDENTIFIER_STRING_STATE_MODULE_SIMULATION)


    # state_module_log_timestamp2, state_module_timestamps2, state_module_states2 = DmLogParser.parse_state_message(
    #     log_content=log_content2,
    #     identifier_str=STATE_IDENTIFIER_STRING_STATE_MODULE_SIMULATION)

    ###########################
    # Plot log analysis
    ###########################


    #ax1 = plt.subplot(311)

    # Plot ego velocity
    actual_v = []
    ego_timestamps = []
    accel = []
    targets = {}
    colors = np.random.random([64, 3])
    for state_message_index in range(len(state_module_states)):
        # Convert log messages to dict
        state_msg = LogMsg.convert_message_to_dict(state_module_states[state_message_index])
        # Deserialize from dict to object
        state = LogMsg.deserialize(class_type=State, message=state_msg)  # type: State
        if state_message_index == 0:
            baseline_timestamp = state.ego_state.timestamp_in_sec
        actual_v.append(state.ego_state.v_x)
        ego_timestamps.append(state.ego_state.timestamp_in_sec - baseline_timestamp)
        accel.append(state.ego_state.acceleration_lon)
        t = state.ego_state.timestamp_in_sec - baseline_timestamp
        for dyn_obj in state.dynamic_objects:
            obj_id = dyn_obj.obj_id
            obj_x, obj_y = dyn_obj.x, dyn_obj.y
            ego_x, ego_y = state.ego_state.x, state.ego_state.y
            obj_range = rng(ego_x, ego_y, obj_x, obj_y)
            target_vel = dyn_obj.v_x
            if obj_id in targets:
                targets[obj_id]["rng"].append(obj_range)
                targets[obj_id]["vel"].append(target_vel)
                targets[obj_id]["t"].append(t)
            else:
                targets[obj_id] = {"rng": [obj_range], "t": [t], "vel": [target_vel]}


    #plt.plot(ego_timestamps, actual_v, '-b')

    # actual_v = []
    # ego_timestamps = []
    # for state_message_index in range(len(state_module_states2)):
    #     # Convert log messages to dict
    #     state_msg = LogMsg.convert_message_to_dict(state_module_states2[state_message_index])
    #     # Deserialize from dict to object
    #     state = LogMsg.deserialize(class_type=State, message=state_msg)  # type: State
    #     if state_message_index == 0:
    #         baseline_timestamp = state.ego_state.timestamp_in_sec
    #     actual_v.append(state.ego_state.v_x)
    #     ego_timestamps.append(state.ego_state.timestamp_in_sec - baseline_timestamp)
    #
    # plt.plot(ego_timestamps, actual_v, '-r')
    # plt.ylabel('$m \cdot s^{-1}$')
    # for tid, trg in targets.items():
    #     clr = colors[tid % 64, :]
    #     plt.plot(trg["t"], trg["vel"], "-", color=clr)


#    ax2 = plt.subplot(312, sharex=ax1)

    # plt.xlabel("time")
    # plt.ylabel("rng [m]")
    # plt.title("Range of objects as a function of time")



    fig, ax_accel = plt.subplots()
    range_ax = ax_accel.twinx()

    # Plot ego acceleration
    # accel = []
    # ego_timestamps = []
    # for state_message_index in range(len(state_module_states)):
    #     # Convert log messages to dict
    #     state_msg = LogMsg.convert_message_to_dict(state_module_states[state_message_index])
    #     # Deserialize from dict to object
    #     state = LogMsg.deserialize(class_type=State, message=state_msg)  # type: State
    #     if state_message_index == 0:
    #         baseline_timestamp = state.ego_state.timestamp_in_sec
    #     accel.append(state.ego_state.acceleration_lon)
    #     ego_timestamps.append(state.ego_state.timestamp_in_sec - baseline_timestamp)


    ax_accel.plot(ego_timestamps, accel, '-b', linewidth=4)
    range_ax.plot(targets[1]["t"], targets[1]["rng"], "-r", linewidth=4)



    # Plot ego acceleration
    # accel = []
    # ego_timestamps = []
    # for state_message_index in range(len(state_module_states2)):
    #     # Convert log messages to dict
    #     state_msg = LogMsg.convert_message_to_dict(state_module_states2[state_message_index])
    #     # Deserialize from dict to object
    #     state = LogMsg.deserialize(class_type=State, message=state_msg)  # type: State
    #     if state_message_index == 0:
    #         baseline_timestamp = state.ego_state.timestamp_in_sec
    #     accel.append(state.ego_state.acceleration_lon)
    #     ego_timestamps.append(state.ego_state.timestamp_in_sec - baseline_timestamp)
    #
    # plt.plot(ego_timestamps, accel, '-r')
    ax_accel.set_ylabel('Acceleration ($m \cdot s^{-2}$)', color='b', fontsize=20, weight='bold')
    range_ax.set_ylabel('Distance to target (meters)', color='r', fontsize=20, weight='bold')
    ax_accel.set_xlabel('Seconds', fontsize=20, weight='bold')

    for label in (ax_accel.get_xticklabels() + ax_accel.get_yticklabels()):

        label.set_fontsize(18)
        label.set_fontweight('bold')

    for label in (range_ax.get_xticklabels() + range_ax.get_yticklabels()):
        label.set_fontsize(18)
        label.set_fontweight('bold')

    for axis in ['top', 'bottom', 'left', 'right']:
        ax_accel.spines[axis].set_linewidth(2)
    # range_ax.set_ylim([0,62])
    ax_accel.set_xlim([5, 23])
    # ax_accel.set_ylim([-3, 0.5])

    plt.axhline(y=60, color='r', linestyle='--')
    plt.axhline(y=37.53, color='b', linestyle='--')

    plt.xlabel('Seconds')

    plt.show()

if __name__ == '__main__':
    main()
