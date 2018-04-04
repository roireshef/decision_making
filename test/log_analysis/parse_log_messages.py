import re
from typing import List

import numpy as np

from decision_making.src.global_constants import LOG_MSG_TRAJECTORY_PLANNER_MISSION_PARAMS, \
    LOG_MSG_BEHAVIORAL_PLANNER_OUTPUT, LOG_MSG_TRAJECTORY_PLANNER_NUM_TRAJECTORIES, LOG_MSG_RECEIVED_STATE, \
    LOG_MSG_STATE_MODULE_PUBLISH_STATE, LOG_MSG_TRAJECTORY_PLANNER_TRAJECTORY_MSG, LOG_MSG_TRAJECTORY_PLANNER_IMPL_TIME, \
    LOG_MSG_BEHAVIORAL_PLANNER_IMPL_TIME, LOG_MSG_BEHAVIORAL_PLANNER_SEMANTIC_ACTION, \
    LOG_MSG_BEHAVIORAL_PLANNER_ACTION_SPEC

# Pattern to find the log timestamp within every log line
LOG_TIME_PATTERN = ": \d+-\d+-\d+ \d+:\d+:\d+,\d+ :"
# Pattern to match the log timestamp components within a log timestamp
LOG_TIME_PARSE_PATTERN = ": (\d+)-(\d+)-(\d+) (\d+):(\d+):(\d+),(\d+) :"

# identifier of the state message received in the TP
STATE_IDENTIFIER_STRING_TP = "trajectory_planning_facade.py.*%s" % LOG_MSG_RECEIVED_STATE
# identifier of the state message received in the BP
STATE_IDENTIFIER_STRING_BP = "behavioral_facade.py.*%s" % LOG_MSG_RECEIVED_STATE
# identifier of the state message send by the state module
STATE_IDENTIFIER_STRING_STATE_MODULE = "state_module.py.*%s" % LOG_MSG_STATE_MODULE_PUBLISH_STATE
STATE_IDENTIFIER_STRING_STATE_MODULE_SIMULATION = "state_module_simulation.py.*%s" % LOG_MSG_STATE_MODULE_PUBLISH_STATE

class DmLogParser:
    """
    Parse different types of messages from log
    """

    @staticmethod
    def parse_state_struct(input_str: str) -> str:
        state_match = re.match(pattern='[^{]*({.*})$', string=input_str)
        return state_match.groups()[0]

    @staticmethod
    def parse_log_timestamp(input_str: str) -> float:
        """
        Parse the timestamp of the log message.
        :param input_str: string containing the timestamp
        :return: seconds passed since the start of the day
        """
        time_match = re.match(pattern=LOG_TIME_PARSE_PATTERN, string=input_str)
        y, mm, d, h, m, s, ms = float(time_match.groups()[0]), float(time_match.groups()[1]), float(
            time_match.groups()[2]), float(time_match.groups()[3]), float(time_match.groups()[4]), float(
            time_match.groups()[5]), float(time_match.groups()[6])

        # return timestamp in [sec]
        return (h * 3600 + m * 60 + s + 0.001 * ms) * 1.0

    @staticmethod
    def parse_state_message(log_content: List[str], identifier_str: str) -> (np.ndarray, np.ndarray, List[str]) :
        """
        Parse the state sent / received in various modules
        :param log_content: list of strings of the log lines
        :param identifier_str: a unique string that identifies the relevant module to parse from
        :return: list of the occurrences of the state inside the log (log timestamp, ego timestamp, string of
          the serialized state)
        """
        log_timestamp = list()
        state_timestamps = list()
        states = list()

        search_pattern = ".*(%s)(.*)%s(.*)" % (LOG_TIME_PATTERN, identifier_str)
        for row in range(len(log_content)):
            state_match = re.match(pattern=search_pattern, string=log_content[row])
            if state_match is not None:
                states.append(DmLogParser.parse_state_struct(input_str=state_match.groups()[2]))
                timestamp_match = re.match(pattern=".*'timestamp': (\d*)", string=state_match.groups()[2])
                timestamp = float(timestamp_match.groups()[0]) * 1E-9
                state_timestamps.append(timestamp)
                log_timestamp.append(DmLogParser.parse_log_timestamp(state_match.groups()[0]))

        state_timestamps = np.array(state_timestamps)
        log_timestamp = np.array(log_timestamp)

        # Reorder by state timestamp
        log_msg_order = np.argsort(state_timestamps)
        state_timestamps = state_timestamps[log_msg_order]
        log_timestamp = log_timestamp[log_msg_order]
        states = [states[x] for x in log_msg_order]

        return log_timestamp, state_timestamps, states

    @staticmethod
    def parse_impl_time(log_content: List[str]) -> (List[float], List[float]):
        """
        Parse the bp / tp implementation times
        :param log_content: list of strings of the log lines
        :return: list of bp implementation time, list of the tp implementation times
        """

        tp_impl_times = list()
        bp_impl_times = list()

        tp_identifier_str = "%s " % LOG_MSG_TRAJECTORY_PLANNER_IMPL_TIME
        tp_search_pattern = ".*(%s)(.*)%s(.*)" % (LOG_TIME_PATTERN, tp_identifier_str)

        bp_identifier_str = "%s " % LOG_MSG_BEHAVIORAL_PLANNER_IMPL_TIME
        bp_search_pattern = ".*(%s)(.*)%s(.*)" % (LOG_TIME_PATTERN, bp_identifier_str)

        for row in range(len(log_content)):

            bp_impl_time_match = re.match(pattern=bp_search_pattern, string=log_content[row])
            if bp_impl_time_match is not None:
                bp_impl_times.append(float(bp_impl_time_match.groups()[2]))

            tp_impl_time_match = re.match(pattern=tp_search_pattern, string=log_content[row])
            if tp_impl_time_match is not None:
                tp_impl_times.append(float(tp_impl_time_match.groups()[2]))


        return bp_impl_times, tp_impl_times

    @staticmethod
    def parse_tp_params(log_content: List[str]) -> (np.ndarray, np.ndarray, List[str]):
        """
        Parse the trajectory params received by the TP
        :param log_content: list of strings of the log lines
        :return: list of the occurrences of the message inside the log (log timestamp, ego timestamp, string of
          the serialized state)
        """
        log_timestamp = list()
        state_timestamps = list()
        states = list()

        identifier_str = "trajectory_planning_facade.py.*%s: " % LOG_MSG_TRAJECTORY_PLANNER_MISSION_PARAMS
        search_pattern = ".*(%s)(.*)%s(.*)" % (LOG_TIME_PATTERN, identifier_str)
        for row in range(len(log_content)):
            state_match = re.match(pattern=search_pattern, string=log_content[row])
            if state_match is not None:
                states.append(DmLogParser.parse_state_struct(input_str=state_match.groups()[2]))
                timestamp_match = re.match(pattern=".*'time': ([0-9\.]*)", string=state_match.groups()[2])
                timestamp = float(timestamp_match.groups()[0])
                state_timestamps.append(timestamp)
                log_timestamp.append(DmLogParser.parse_log_timestamp(state_match.groups()[0]))

        state_timestamps = np.array(state_timestamps)
        log_timestamp = np.array(log_timestamp)

        # Reorder by log timestamp
        log_msg_order = np.argsort(log_timestamp)
        state_timestamps = state_timestamps[log_msg_order]
        log_timestamp = log_timestamp[log_msg_order]
        states = [states[x] for x in log_msg_order]

        return log_timestamp, state_timestamps, states

    @staticmethod
    def parse_tp_output(log_content: List[str]) -> (np.ndarray, np.ndarray, List[str]):
        """
        Parse the trajectory sent by the TP
        :param log_content: list of strings of the log lines
        :return: list of the occurrences of the message inside the log (log timestamp, ego timestamp, string of
          the serialized state)
        """
        log_timestamp = list()
        output_timestamps = list()
        messages = list()

        identifier_str = "%s: " % LOG_MSG_TRAJECTORY_PLANNER_TRAJECTORY_MSG

        search_pattern = ".*(%s)(.*)%s(.*)" % (LOG_TIME_PATTERN, identifier_str)

        for row in range(len(log_content)):
            state_match = re.match(pattern=search_pattern, string=log_content[row])
            if state_match is not None:
                messages.append(state_match.groups()[2])
                # TODO: parse timestamp. currently this message isn't written properly to log
                # timestamp_match = re.match(pattern=".*'time': ([0-9\.]*)", string=state_match.groups()[2])
                # timestamp = float(timestamp_match.groups()[0])
                timestamp = 0.0
                output_timestamps.append(timestamp)
                log_timestamp.append(DmLogParser.parse_log_timestamp(state_match.groups()[0]))

        output_timestamps = np.array(output_timestamps)
        log_timestamp = np.array(log_timestamp)

        # Reorder by log timestamp
        log_msg_order = np.argsort(log_timestamp)
        output_timestamps = output_timestamps[log_msg_order]
        log_timestamp = log_timestamp[log_msg_order]
        messages = [messages[x] for x in log_msg_order]

        return log_timestamp, output_timestamps, messages

    @staticmethod
    def parse_bp_action(log_content: List[str]) -> (np.ndarray, np.ndarray, List[str]):
        """
        Parse the output sent by the BP
        :param log_content: list of strings of the log lines
        :return: list of the occurrences of the message inside the log (log timestamp, ego timestamp, string of
          the serialized message)
        """
        semantic_action_messages = list()
        semantic_action_log_timestamp = list()
        action_spec_messages = list()
        action_spec_log_timestamp = list()

        semantic_action_identifier_str = "%s " % LOG_MSG_BEHAVIORAL_PLANNER_SEMANTIC_ACTION
        action_spec_identifier_str = "%s " % LOG_MSG_BEHAVIORAL_PLANNER_ACTION_SPEC
        semantic_action_search_pattern = ".*(%s)(.*)%s(.*)" % (LOG_TIME_PATTERN, semantic_action_identifier_str)
        action_spec_search_pattern = ".*(%s)(.*)%s(.*)" % (LOG_TIME_PATTERN, action_spec_identifier_str)
        for row in range(len(log_content)):
            semantic_action_match = re.match(pattern=semantic_action_search_pattern, string=log_content[row])
            action_spec_match = re.match(pattern=action_spec_search_pattern, string=log_content[row])
            if semantic_action_match is not None:
                semantic_action_messages.append(semantic_action_match.groups()[2])
                semantic_action_log_timestamp.append(DmLogParser.parse_log_timestamp(semantic_action_match.groups()[0]))
            if action_spec_match is not None:
                action_spec_messages.append(action_spec_match.groups()[2])
                action_spec_log_timestamp.append(DmLogParser.parse_log_timestamp(action_spec_match.groups()[0]))

        semantic_action_log_timestamp = np.array(semantic_action_log_timestamp)
        action_spec_log_timestamp = np.array(action_spec_log_timestamp)

        # Reorder by log timestamp
        semantic_action_log_msg_order = np.argsort(semantic_action_log_timestamp)
        semantic_action_log_timestamp = semantic_action_log_timestamp[semantic_action_log_msg_order]
        semantic_action_messages = [semantic_action_messages[x] for x in semantic_action_log_msg_order]

        action_spec_log_msg_order = np.argsort(action_spec_log_timestamp)
        action_spec_log_timestamp = action_spec_log_timestamp[action_spec_log_msg_order]
        action_spec_messages = [action_spec_messages[x] for x in action_spec_log_msg_order]

        return semantic_action_log_timestamp, semantic_action_messages, action_spec_log_timestamp, action_spec_messages

    @staticmethod
    def parse_bp_output(log_content: List[str]) -> (np.ndarray, np.ndarray, List[str]):
        """
        Parse the output sent by the BP
        :param log_content: list of strings of the log lines
        :return: list of the occurrences of the message inside the log (log timestamp, ego timestamp, string of
          the serialized message)
        """
        log_timestamp = list()
        state_timestamps = list()
        messages = list()

        identifier_str = "%s " % LOG_MSG_BEHAVIORAL_PLANNER_OUTPUT
        search_pattern = ".*(%s)(.*)%s(.*)" % (LOG_TIME_PATTERN, identifier_str)
        for row in range(len(log_content)):
            state_match = re.match(pattern=search_pattern, string=log_content[row])
            if state_match is not None:
                messages.append(state_match.groups()[2])
                # TODO: parse timestamp. currently this message isn't written properly to log
                # timestamp_match = re.match(pattern=".*'time': ([0-9\.]*)", string=state_match.groups()[2])
                # timestamp = float(timestamp_match.groups()[0])
                timestamp = 0.0
                state_timestamps.append(timestamp)
                log_timestamp.append(DmLogParser.parse_log_timestamp(state_match.groups()[0]))

        state_timestamps = np.array(state_timestamps)
        log_timestamp = np.array(log_timestamp)

        # Reorder by log timestamp
        log_msg_order = np.argsort(log_timestamp)
        state_timestamps = state_timestamps[log_msg_order]
        log_timestamp = log_timestamp[log_msg_order]
        messages = [messages[x] for x in log_msg_order]

        return log_timestamp, state_timestamps, messages

    @staticmethod
    def parse_no_valid_trajectories_message(log_content: List[str]) -> (np.ndarray):
        """
        Parse the times where no trajectories were found
        :param log_content: list of strings of the log lines
        :return: list of log timestamps
        """
        log_timestamp = list()

        identifier_str = LOG_MSG_TRAJECTORY_PLANNER_NUM_TRAJECTORIES % 0
        search_pattern = ".*(%s)(.*)%s(.*)" % (LOG_TIME_PATTERN, identifier_str)
        for row in range(len(log_content)):
            state_match = re.match(pattern=search_pattern, string=log_content[row])
            if state_match is not None:
                log_timestamp.append(DmLogParser.parse_log_timestamp(state_match.groups()[0]))

        log_timestamp = np.array(log_timestamp)

        # Reorder by log timestamp
        log_msg_order = np.argsort(log_timestamp)
        log_timestamp = log_timestamp[log_msg_order]

        return log_timestamp


    @staticmethod
    def parse_control_error(log_content: List[str]) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        Parse the control errors
        :param log_content: list of strings of the log lines
        :return: list of the control errors (log timestamp, longitudinal error in [m], lateral error in [m])
        """
        log_timestamp = list()
        control_errors_lon = list()
        control_errors_lat = list()

        identifier_str = "trajectory_planning_facade.py:.*lon_lat_errors: \[\s+([0-9\.]+)\s+([0-9\.]+).*"
        search_pattern = ".*(%s).*%s" % (LOG_TIME_PATTERN, identifier_str)
        for row in range(len(log_content)):
            state_match = re.match(pattern=search_pattern, string=log_content[row])
            if state_match is not None:
                log_timestamp.append(DmLogParser.parse_log_timestamp(state_match.groups()[0]))
                lon_error, lat_error = state_match.groups()[1], state_match.groups()[2]
                control_errors_lon.append(lon_error)
                control_errors_lat.append(lat_error)

        control_errors_lon = np.array(control_errors_lon)
        control_errors_lat = np.array(control_errors_lat)
        log_timestamp = np.array(log_timestamp)

        # Reorder by log timestamp
        log_msg_order = np.argsort(log_timestamp)
        log_timestamp = log_timestamp[log_msg_order]
        control_errors_lon = control_errors_lon[log_msg_order]
        control_errors_lat = control_errors_lat[log_msg_order]

        return log_timestamp, control_errors_lon, control_errors_lat

