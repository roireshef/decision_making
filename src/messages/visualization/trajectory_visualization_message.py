from typing import List

import numpy as np

from Rte_Types import TsSYSTrajectoryVisualization
from Rte_Types.sub_structures import TsSYSDataTrajectoryVisualization
from Rte_Types.sub_structures.TsSYS_PredictionsVisualization import TsSYSPredictionsVisualization
from decision_making.src.global_constants import PUBSUB_MSG_IMPL
from decision_making.src.messages.scene_common_messages import Header


class PredictionsVisualization(PUBSUB_MSG_IMPL):
    def __init__(self, e_object_id: int, a_predictions: np.array):
        """
        The class contains predicted locations for single dynamic object
        :param e_object_id:
        :param a_predictions: predicted 2D locations of the object
        """
        pass

    def serialize(self) -> TsSYSPredictionsVisualization:
        return TsSYSPredictionsVisualization()

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSPredictionsVisualization):
        pass


class DataTrajectoryVisualization(PUBSUB_MSG_IMPL):
    def __init__(self, a_trajectories: np.ndarray, as_actors_predictions: List[PredictionsVisualization],
                 e_recipe_description: str):
        """
        Message that holds debug results of WerlingPlanner to be broadcasted to the visualizer
        :param a_trajectories: 3D array of trajectories: num_trajectories x trajectory_length x 2
        :param as_actors_predictions: list of classes of type PredictionsVisualization per dynamic object.
                Each class instance contains predictions for the dynamic object.
        :param e_recipe_description: String for semantic meaning of action. For example:
                                                            "static action to the left with 50 km/h".
        """
        pass

    def serialize(self) -> TsSYSDataTrajectoryVisualization:
        return TsSYSDataTrajectoryVisualization()

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSDataTrajectoryVisualization):
        pass


class TrajectoryVisualizationMsg(PUBSUB_MSG_IMPL):
    def __init__(self, s_Header: Header, s_Data: DataTrajectoryVisualization):
        pass

    def serialize(self) -> TsSYSTrajectoryVisualization:
        return TsSYSTrajectoryVisualization()

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSTrajectoryVisualization):
        pass
