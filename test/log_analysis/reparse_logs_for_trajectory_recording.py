import sys
import json

from decision_making.src.messages.trajectory_parameters import TrajectoryParams
from decision_making.src.messages.trajectory_plan_message import TrajectoryPlanMsg
from decision_making.src.planning.types import C_V

from decision_making.src.state.state import State
from decision_making.test.log_analysis.log_messages import LogMsg
from decision_making.test.log_analysis.parse_log_messages import STATE_IDENTIFIER_STRING_BP, \
    STATE_IDENTIFIER_STRING_TP, STATE_IDENTIFIER_STRING_STATE_MODULE, DmLogParser

import matplotlib.pyplot as plt


# LOG_PATH_FOR_ANALYSIS = '/home/max/AV_Log_dm_main_test-2017_12_12-10_19.log'
LOG_PATH_FOR_ANALYSIS = '/home/nz2v30/recordings/2018_03_12_12_34_Proving_Grounds_Daytime/logs/AV_Log_dm_main.log'
#LOG_PATH_FOR_ANALYSIS = '/data/recordings/cdrive/Database/2018_02_19/2018_02_19_16_56_Proving_Grounds_-_Low_Light/AV_Log_dm_main.log'
# LOG_PATH_FOR_ANALYSIS = 'AV_Log_rbcar0.log'
TARGET_LOG_TIME = 57906.0
TARGET_OUTPUT_DIR = '/home/nz2v30/reparser_output/'

def main():
    filename = LOG_PATH_FOR_ANALYSIS if ('log_filename' not in sys.argv) else sys.argv['log_filename']

    f = open(file=filename, mode='r')
    log_content = f.readlines()

    ###########################
    # Parse relevant messages
    ###########################

    # State module
    state_module_log_timestamp, state_module_timestamps, state_module_states = DmLogParser.parse_state_message(
        log_content=log_content,
        identifier_str=STATE_IDENTIFIER_STRING_STATE_MODULE)

    # BP module
    bp_state_log_timestamp, bp_state_timestamps, bp_states = DmLogParser.parse_state_message(log_content=log_content,
                                                                                             identifier_str=STATE_IDENTIFIER_STRING_BP)
    bp_module_log_timestamp, bp_module_timestamps, bp_module_states = DmLogParser.parse_bp_output(
        log_content=log_content)

    # TP module
    tp_state_log_timestamp, tp_state_timestamps, tp_states = DmLogParser.parse_state_message(log_content=log_content,
                                                                                             identifier_str=STATE_IDENTIFIER_STRING_TP)
    tp_module_log_timestamp, tp_module_timestamps, tp_module_states = DmLogParser.parse_tp_params(
        log_content=log_content)
    no_valid_trajectories_log_timestamp = DmLogParser.parse_no_valid_trajectories_message(log_content=log_content)

    tp_plan_log_timestamp, tp_plan_timestamps, tp_plans = DmLogParser.parse_tp_output(log_content=log_content)

    bp_impl_times, tp_impl_times = DmLogParser.parse_impl_time(log_content=log_content)

    ###########################
    # Send messages to module
    ###########################


    with open("av_log_states.json", 'w') as file:

        for time in bp_impl_times:
            time_dict = dict()
            time_dict['bp_impl_time'] = time
            file.write(json.dumps(time_dict))
            file.write("\n")

        for time in tp_impl_times:
            time_dict = dict()
            time_dict['tp_impl_time'] = time
            file.write(json.dumps(time_dict))
            file.write("\n")

        for tp_plan_message_index in range(len(tp_plans)):
            # Convert log messages to dict
            plan_msg = LogMsg.convert_message_to_dict(tp_plans[tp_plan_message_index])
            plan = LogMsg.deserialize(class_type=TrajectoryPlanMsg, message=plan_msg)
            plan_serialized = plan.to_dict()
            plan_serialized['msg_type'] = "tp_plan"
            plan_serialized['log_timestamp'] = tp_plan_log_timestamp[tp_plan_message_index]
            file.write(json.dumps(plan_serialized))
            file.write("\n")


        timestamps = []
        desired_v = []

        for tp_plan_message_index in range(len(tp_plans)):
            # Convert log messages to dict
            plan_msg = LogMsg.convert_message_to_dict(tp_plans[tp_plan_message_index])
            plan = LogMsg.deserialize(class_type=TrajectoryPlanMsg, message=plan_msg) #type: TrajectoryPlanMsg
            timestamps.append(plan.timestamp)
            desired_v.append(plan.trajectory[0, C_V])

        plt.plot(timestamps, desired_v, '-r')



        ego_timestamps = []
        actual_v = []
        prev_timestamps = {}
        agent_file_dict = {}
        minimal_dyn_obj_timestamp = float("inf")
        ego_in_minimal_dyn_obj_timestamp = None
        for state_message_index in range(len(state_module_states)):
            # Convert log messages to dict
            state_msg = LogMsg.convert_message_to_dict(state_module_states[state_message_index])

            # Deserialize from dict to object
            state = LogMsg.deserialize(class_type=State, message=state_msg) # type: State
            ego_timestamps.append(state.ego_state.timestamp)
            actual_v.append(state.ego_state.v_x)
            state_serialized = state.to_dict()
            state_serialized['msg_type'] = "state_output"
            state_serialized['log_timestamp'] = state_module_log_timestamp[state_message_index]
            file.write(json.dumps(state_serialized))
            file.write("\n")

            # for object1 in tp_params.dynamic_objects:
            #     file.write(str(object1.timestamp)+","+str(object1.obj_id)+","+str(object1.v_x)+","+str(object1.v_y)+","+str(tp_params.ego_state.v_x)+","+str(tp_params.ego_state.timestamp))
            #     file.write("\n")

            for dyn_obj in state.dynamic_objects:
                if dyn_obj.obj_id not in prev_timestamps.keys():
                    prev_timestamps[dyn_obj.obj_id] = 0
                    agent_file_dict[dyn_obj.obj_id] = open(TARGET_OUTPUT_DIR + "car_" + str(dyn_obj.obj_id) + ".txt", 'w')

                if dyn_obj.timestamp_in_sec > prev_timestamps[dyn_obj.obj_id]:
                    out_str = str(dyn_obj.timestamp_in_sec) + ", " + str(dyn_obj.x) + ", " + str(dyn_obj.y) + ", " + \
                              str(dyn_obj.yaw) + "\n"
                    agent_file_dict[dyn_obj.obj_id].write(out_str)
                    prev_timestamps[dyn_obj.obj_id] = dyn_obj.timestamp_in_sec

                if dyn_obj.timestamp_in_sec < minimal_dyn_obj_timestamp:
                    minimal_dyn_obj_timestamp = dyn_obj.timestamp_in_sec
                    ego_in_minimal_dyn_obj_timestamp = state.ego_state

        # done with all states, creating ego description
        f = open(TARGET_OUTPUT_DIR + "ego.txt", 'w')
        out_str = str(ego_in_minimal_dyn_obj_timestamp.timestamp_in_sec) + ", " +\
                  str(ego_in_minimal_dyn_obj_timestamp.x) + ", " + \
                  str(ego_in_minimal_dyn_obj_timestamp.y) + ", " + \
                  str(ego_in_minimal_dyn_obj_timestamp.yaw) + "\n"
        f.write(out_str)
        f.close()




        plt.plot(ego_timestamps, actual_v, '-b')
        plt.show()
        print("finish")


    for f in agent_file_dict.values():
        f.close()





        # for tp_params_message_index in range(len(tp_module_states)):
        #     # Convert log messages to dict
        #     tp_params_msg = LogMsg.convert_message_to_dict(tp_module_states[tp_params_message_index])
        #
        #     # Deserialize from dict to object
        #     tp_params = LogMsg.deserialize(class_type=TrajectoryParams, message=tp_params_msg)
        #     tp_params_serialized = tp_params.to_dict()
        #     tp_params_serialized['msg_type'] = "tp_input_params"
        #     tp_params_serialized['log_timestamp'] = tp_module_log_timestamp[tp_params_message_index]
        #     file.write(json.dumps(tp_params_serialized))
        #     file.write("\n")
        #
        # for tp_state_message_index in range(len(tp_states)):
        #     tp_state_msg = LogMsg.convert_message_to_dict(tp_states[tp_state_message_index])
        #     tp_state = LogMsg.deserialize(class_type=State, message=tp_state_msg)
        #
        #     # Serialize object to PubSub dict
        #     tp_state_serialized = tp_state.to_dict()
        #     tp_state_serialized['msg_type'] = "tp_input_state"
        #     tp_state_serialized['log_timestamp'] = tp_state_log_timestamp[tp_state_message_index]
        #     file.write(json.dumps(tp_state_serialized))
        #     file.write("\n")
        #
        # for bp_message_index in range(len(bp_states)):
        #     bp_state_msg = LogMsg.convert_message_to_dict(bp_states[bp_message_index])
        #     bp_state = LogMsg.deserialize(class_type=State, message=bp_state_msg)
        #
        #     # Serialize object to PubSub dict
        #     bp_state_serialized = bp_state.to_dict()
        #     bp_state_serialized['msg_type'] = "bp_input_state"
        #     bp_state_serialized['log_timestamp'] = bp_state_log_timestamp[bp_message_index]
        #     file.write(json.dumps(bp_state_serialized))
        #     file.write("\n")
        #
        # for bp_params_message_index in range(len(bp_module_states)):
        #     # Convert log messages to dict
        #     bp_params_msg = LogMsg.convert_message_to_dict(bp_module_states[bp_params_message_index])
        #
        #     # Deserialize from dict to object
        #     bp_params = LogMsg.deserialize(class_type=TrajectoryParams, message=bp_params_msg)
        #     bp_params_serialized = bp_params.to_dict()
        #     bp_params_serialized['msg_type'] = "bp_output_params"
        #     bp_params_serialized['log_timestamp'] = bp_module_log_timestamp[bp_params_message_index]
        #     file.write(json.dumps(bp_params_serialized))
        #     file.write("\n")


if __name__ == '__main__':
    main()