import sys
import numpy as np
import matplotlib.pyplot as plt
import re

from decision_making.src.global_constants import BEHAVIORAL_PLANNING_MODULE_PERIOD
from decision_making.src.messages.trajectory_plan_message import TrajectoryPlanMsg
from decision_making.src.planning.behavioral.policies.semantic_actions_policy import SemanticActionType
from decision_making.src.planning.types import C_V
from decision_making.src.state.state import State
from decision_making.test.log_analysis.log_messages import LogMsg
from decision_making.test.log_analysis.parse_log_messages import DmLogParser, \
    STATE_IDENTIFIER_STRING_BP, STATE_IDENTIFIER_STRING_TP, STATE_IDENTIFIER_STRING_STATE_MODULE, \
    STATE_IDENTIFIER_STRING_STATE_MODULE_SIMULATION

#LOG_PATH_FOR_ANALYSIS = '/local/flash/recordings/2018_03_26/2018_03_26_16_28/AV_Log_dm_main.log'
LOG_PATH_FOR_ANALYSIS = '/home/kz430x/BitBucket/spav/logs/AV_Log_rbcar0.log'

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


    ax1 = plt.subplot(511)

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

    plt.plot(ego_timestamps, actual_v, '-b')

    plt.ylabel('$m \cdot s^{-1}$')
    for tid, trg in targets.items():
        clr = colors[tid % 64, :]
        plt.plot(trg["t"], trg["vel"], "-", color=clr)


    ax2 = plt.subplot(512, sharex=ax1)
    for tid, trg in targets.items():
        clr = colors[tid % 64, :]
        plt.plot(trg["t"], trg["rng"], "-", color=clr)

    plt.xlabel("time")
    plt.ylabel("rng [m]")
    plt.title("Range of objects as a function of time")

    ax3 = plt.subplot(513, sharex=ax2)

    plt.plot(ego_timestamps, accel, '-b')

    plt.ylabel('$m \cdot s^{-2}$')


    # # BP module action specification time
    # _, _, bp_msgs_content = DmLogParser.parse_bp_output(log_content=log_content)
    # bp_t_specifications = []
    # bp_ego_timestamps = []
    # for bp_message_index in range(len(bp_msgs_content)):
    #     bp_msg_dict = LogMsg.convert_message_to_dict(bp_msgs_content[bp_message_index])
    #     specify_global_t = bp_msg_dict['time']
    #     planning_t = BEHAVIORAL_PLANNING_MODULE_PERIOD * bp_message_index
    #     bp_ego_timestamps.append(planning_t)
    #     bp_t_specifications.append(specify_global_t - planning_t)

    # BP type of semantic action
    _, semnatic_actions, _, action_specs = DmLogParser.parse_bp_action(log_content=log_content)
    bp_t_specifications = []
    bp_semantic_action_type = []
    bp_ego_timestamps = []
    for bp_message_index in range(len(action_specs)):
        bp_action_dict = LogMsg.convert_message_to_dict(semnatic_actions[bp_message_index])
        bp_action_specs_dict = LogMsg.convert_message_to_dict(action_specs[bp_message_index])

        planning_t = BEHAVIORAL_PLANNING_MODULE_PERIOD * bp_message_index
        bp_ego_timestamps.append(planning_t)

        bp_semantic_action_type.append(int(
            re.match(pattern='.*: (\d+).*', string=bp_action_dict['action_type']).groups()[0]))
        bp_t_specifications.append(bp_action_specs_dict['t'])

    ax4 = plt.subplot(514, sharex=ax3)

    plt.plot(bp_ego_timestamps, bp_t_specifications)
    plt.ylabel('$t (specify) [sec.]$')

    ax5 = plt.subplot(515, sharex=ax4)

    plt.plot(bp_ego_timestamps, bp_semantic_action_type)
    plt.ylabel('action type index\n1 - follow vehicle\n2 - follow lane)')

    plt.xlabel('$timestamp [sec.]$')

    plt.show()

if __name__ == '__main__':
    main()