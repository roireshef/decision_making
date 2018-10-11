from logging import Logger

from decision_making.src.state.state import State


class BehavioralState:
    @classmethod
    def create_from_state(cls, state: State, map_object: MapObject,logger: Logger):
        """
        This method updates the behavioral state according to the new world state and navigation plan.
         It fetches relevant features that will be used for the decision-making process.
        :param state: new world state
        :param logger
        :return: a new and updated BehavioralState
        """
        pass