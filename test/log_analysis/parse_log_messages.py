import re
from typing import List

import numpy as np

from decision_making.src.global_constants import LOG_MSG_TRAJECTORY_PLANNER_MISSION_PARAMS, \
    LOG_MSG_BEHAVIORAL_PLANNER_OUTPUT, LOG_MSG_TRAJECTORY_PLANNER_NUM_TRAJECTORIES, LOG_MSG_RECEIVED_STATE, \
    LOG_MSG_STATE_MODULE_PUBLISH_STATE

LOG_TIME_PATTERN = ": \d+-\d+-\d+ \d+:\d+:\d+,\d+ :"
LOG_TIME_PARSE_PATTERN = ": (\d+)-(\d+)-(\d+) (\d+):(\d+):(\d+),(\d+) :"


STATE_IDENTIFIER_STRING_TP = "trajectory_planning_facade.py.*%s" % LOG_MSG_RECEIVED_STATE
STATE_IDENTIFIER_STRING_BP = "behavioral_facade.py.*%s" % LOG_MSG_RECEIVED_STATE
STATE_IDENTIFIER_STRING_STATE_MODULE = "state_module.py.*%s" % LOG_MSG_STATE_MODULE_PUBLISH_STATE


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
        time_match = re.match(pattern=LOG_TIME_PARSE_PATTERN, string=input_str)
        y, mm, d, h, m, s, ms = float(time_match.groups()[0]), float(time_match.groups()[1]), float(
            time_match.groups()[2]), float(time_match.groups()[3]), float(time_match.groups()[4]), float(
            time_match.groups()[5]), float(time_match.groups()[6])

        # return timestamp in [sec]
        return (h * 3600 + m * 60 + s + 0.001 * ms) * 1.0

    @staticmethod
    def parse_state_message(log_content: List[str], identifier_str: str) -> (np.ndarray, np.ndarray, List[str]) :
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

        # Reorder by log timestamp
        log_msg_order = np.argsort(log_timestamp)
        state_timestamps = state_timestamps[log_msg_order]
        log_timestamp = log_timestamp[log_msg_order]
        states = [states[x] for x in log_msg_order]

        return log_timestamp, state_timestamps, states

    @staticmethod
    def parse_tp_params(log_content: List[str]) -> (np.ndarray, np.ndarray, List[str]):
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
    def parse_bp_output(log_content: List[str]) -> (np.ndarray, np.ndarray, List[str]):
        log_timestamp = list()
        state_timestamps = list()
        states = list()

        identifier_str = "%s " % LOG_MSG_BEHAVIORAL_PLANNER_OUTPUT
        search_pattern = ".*(%s)(.*)%s(.*)" % (LOG_TIME_PATTERN, identifier_str)
        for row in range(len(log_content)):
            state_match = re.match(pattern=search_pattern, string=log_content[row])
            if state_match is not None:
                states.append(state_match.groups()[2])
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
        states = [states[x] for x in log_msg_order]

        return log_timestamp, state_timestamps, states

    @staticmethod
    def parse_no_valid_trajectories_message(log_content: List[str]) -> (np.ndarray):
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

