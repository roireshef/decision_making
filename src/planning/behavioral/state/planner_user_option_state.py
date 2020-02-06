from typing import Tuple
from decision_making.src.messages.serialization import PUBSUB_MSG_IMPL


import numpy as np
from decision_making.src.global_constants import GAP_SETTING_HEADWAY, GAP_SETTING_COMFORT_HDW_MAX, \
GAP_SETTING_COMFORT_HDW_MIN
from decision_making.src.messages.gap_setting_message import GapSetting, GapSettingState



class PlannerUserOptionState(PUBSUB_MSG_IMPL):

    def __init__(self):
        """
        Holds user option states
        """
        self.previous_gap_setting_state = None
        self.gap_setting_state = None
        self.gap_setting_change_time = np.inf
        self.current_time = np.inf

    def update(self, timestamp_in_sec: float, gap_setting: GapSetting):
        """
        Update's the user settable options
        :param timestamp_in_sec: current timestamp from scene dynamic
        :param gap_setting: received gap setting message
        :return:
        """
        self.current_time = timestamp_in_sec
        gap_setting_state = gap_setting.s_Data.e_e_gap_setting_state

        # update gap settings if changed
        if gap_setting_state != self.gap_setting_state:
            self.previous_gap_setting_state = self.gap_setting_state
            if not self.previous_gap_setting_state:
                self.previous_gap_setting_state = self.gap_setting_state

            self.gap_setting_state = gap_setting_state
            self.gap_setting_change_time = self.current_time

    def get_headway(self) -> Tuple[float, float, float]:
        """
        Gets headway as specified by the user settings.
        A delay is forced before a change will take effect.
        When changing between settings, the headway will gradually change to the desired one over a set time.
        :return: (gap_setting_headway, gap_setting_headway_min, gap_setting_headway_max)
        """

        # Return default value if object was not initialized
        if self.gap_setting_state is None or self.previous_gap_setting_state is None:
            headway = GAP_SETTING_HEADWAY[GapSettingState.CeSYS_e_Medium.value]
            comfort_hdw_min = GAP_SETTING_COMFORT_HDW_MIN[GapSettingState.CeSYS_e_Medium.value]
            comfort_hdw_max = GAP_SETTING_COMFORT_HDW_MAX[GapSettingState.CeSYS_e_Medium.value]
            return (headway, headway + comfort_hdw_min, headway + comfort_hdw_max)

        headway = GAP_SETTING_HEADWAY[self.gap_setting_state.value]
        comfort_hdw_min = GAP_SETTING_COMFORT_HDW_MIN[self.gap_setting_state.value]
        comfort_hdw_max = GAP_SETTING_COMFORT_HDW_MAX[self.gap_setting_state.value]

        return (headway, headway + comfort_hdw_min, headway + comfort_hdw_max)




