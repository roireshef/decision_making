from logging import Logger
from typing import Dict

from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.planning.behavioral.data_objects import RelativeLane
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GeneralizedFrenetSerretFrame
from decision_making.src.state.state import State


class BehavioralState:
    @classmethod
    def create_from_state(cls, state: State, unified_frames: Dict[RelativeLane, GeneralizedFrenetSerretFrame], logger: Logger):
        """
        This method updates the behavioral state according to the new world state and navigation plan.
         It fetches relevant features that will be used for the decision-making process.
        :param state: new world state
        :param unified_frames: dictionary from RelativeLane to the relevant generalized frenet frame
        :param logger
        :return: a new and updated BehavioralState
        """
        pass