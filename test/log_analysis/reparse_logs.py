import sys
import json

from decision_making.src.messages.class_serialization import ClassSerializer
from decision_making.src.messages.trajectory_parameters import TrajectoryParams
from decision_making.src.messages.trajectory_plan_message import TrajectoryPlanMsg
from decision_making.src.planning.types import C_V

from decision_making.src.state.state import State
from decision_making.test.log_analysis.parse_log_messages import STATE_IDENTIFIER_STRING_BP, \
    STATE_IDENTIFIER_STRING_TP, STATE_IDENTIFIER_STRING_STATE_MODULE, DmLogParser

import matplotlib.pyplot as plt


# Log filename
LOG_PATH_FOR_ANALYSIS = '/data/recordings/cdrive/Database/2018_02_19/2018_02_19_15_56_Proving_Grounds_-_Daytime/AV_Log_dm_main_2018-02-19_15-58-06.log'


def main():
    """
    Use the log parsing tool to parse relevant messages and rewrite them as a JSON file.
    """

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

        # Write BP implementation times
        for time in bp_impl_times:
            time_dict = dict()
            time_dict['bp_impl_time'] = time
            file.write(json.dumps(time_dict))
            file.write("\n")

        # TP implementation times
        for time in tp_impl_times:
            time_dict = dict()
            time_dict['tp_impl_time'] = time
            file.write(json.dumps(time_dict))
            file.write("\n")

        # TP plan
        for tp_plan_message_index in range(len(tp_plans)):
            # Convert log messages to dict
            plan_msg = ClassSerializer.convert_message_to_dict(tp_plans[tp_plan_message_index])
            plan = ClassSerializer.deserialize(class_type=TrajectoryPlanMsg, message=plan_msg)
            plan_serialized = plan.to_dict()
            plan_serialized['msg_type'] = "tp_plan"
            plan_serialized['log_timestamp'] = tp_plan_log_timestamp[tp_plan_message_index]
            file.write(json.dumps(plan_serialized))
            file.write("\n")

        # State module output
        for state_message_index in range(len(state_module_states)):
             # Convert log messages to dict
            state_msg = ClassSerializer.convert_message_to_dict(state_module_states[state_message_index])

            # Deserialize from dict to object
            state = ClassSerializer.deserialize(class_type=State, message=state_msg) # type: State
            state_serialized = state.to_dict()
            state_serialized['msg_type'] = "state_output"
            state_serialized['log_timestamp'] = state_module_log_timestamp[state_message_index]
            file.write(json.dumps(state_serialized))
            file.write("\n")

        # TP input: TP params
        for tp_params_message_index in range(len(tp_module_states)):
            # Convert log messages to dict
            tp_params_msg = ClassSerializer.convert_message_to_dict(tp_module_states[tp_params_message_index])

            # Deserialize from dict to object
            tp_params = ClassSerializer.deserialize(class_type=TrajectoryParams, message=tp_params_msg)
            tp_params_serialized = tp_params.to_dict()
            tp_params_serialized['msg_type'] = "tp_input_params"
            tp_params_serialized['log_timestamp'] = tp_module_log_timestamp[tp_params_message_index]
            file.write(json.dumps(tp_params_serialized))
            file.write("\n")

        # TP input: state
        for tp_state_message_index in range(len(tp_states)):
            tp_state_msg = ClassSerializer.convert_message_to_dict(tp_states[tp_state_message_index])
            tp_state = ClassSerializer.deserialize(class_type=State, message=tp_state_msg)

            # Serialize object to PubSub dict
            tp_state_serialized = tp_state.to_dict()
            tp_state_serialized['msg_type'] = "tp_input_state"
            tp_state_serialized['log_timestamp'] = tp_state_log_timestamp[tp_state_message_index]
            file.write(json.dumps(tp_state_serialized))
            file.write("\n")

        # BP input: state
        for bp_message_index in range(len(bp_states)):
            bp_state_msg = ClassSerializer.convert_message_to_dict(bp_states[bp_message_index])
            bp_state = ClassSerializer.deserialize(class_type=State, message=bp_state_msg)

            # Serialize object to PubSub dict
            bp_state_serialized = bp_state.to_dict()
            bp_state_serialized['msg_type'] = "bp_input_state"
            bp_state_serialized['log_timestamp'] = bp_state_log_timestamp[bp_message_index]
            file.write(json.dumps(bp_state_serialized))
            file.write("\n")

        # BP output: TP params
        for bp_params_message_index in range(len(bp_module_states)):
            # Convert log messages to dict
            bp_params_msg = ClassSerializer.convert_message_to_dict(bp_module_states[bp_params_message_index])

            # Deserialize from dict to object
            bp_params = ClassSerializer.deserialize(class_type=TrajectoryParams, message=bp_params_msg)
            bp_params_serialized = bp_params.to_dict()
            bp_params_serialized['msg_type'] = "bp_output_params"
            bp_params_serialized['log_timestamp'] = bp_module_log_timestamp[bp_params_message_index]
            file.write(json.dumps(bp_params_serialized))
            file.write("\n")

        print("finish")


if __name__ == '__main__':
    main()
