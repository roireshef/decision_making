from typing import List

import numpy as np

from common_data.interface.Rte_Types.python.sub_structures import TsSYSTrajectoryVisualization
from common_data.interface.Rte_Types.python.sub_structures import TsSYSDataTrajectoryVisualization
from common_data.interface.Rte_Types.python.sub_structures import TsSYSPredictionsVisualization
from decision_making.src.global_constants import PUBSUB_MSG_IMPL
from decision_making.src.messages.scene_common_messages import Header


class PredictionsVisualization(PUBSUB_MSG_IMPL):
    def __init__(self, e_object_id: int, a_predictions: np.array):
        """
        The class contains predicted locations for single dynamic object
        :param e_object_id:
        :param a_predictions: predicted 2D locations of the object
        """
        self.e_object_id = e_object_id
        self.a_predictions = a_predictions

    def serialize(self) -> TsSYSPredictionsVisualization:
        pubsub_msg = TsSYSPredictionsVisualization()

        pubsub_msg.e_object_id = self.e_object_id
        pubsub_msg.e_Cnt_num_predictions = self.a_predictions.shape[0]
        pubsub_msg.a_predictions = self.a_predictions

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSPredictionsVisualization):
        return cls(pubsubMsg.e_object_id, pubsubMsg.a_predictions[:pubsubMsg.e_Cnt_num_predictions])


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
        self.a_trajectories = a_trajectories
        self.as_actors_predictions = as_actors_predictions
        self.e_recipe_description = e_recipe_description

    def serialize(self) -> TsSYSDataTrajectoryVisualization:
        pubsub_msg = TsSYSDataTrajectoryVisualization()

        pubsub_msg.e_Cnt_num_points_in_trajectory = self.a_trajectories.shape[1]
        pubsub_msg.e_Cnt_num_trajectories = self.a_trajectories.shape[0]
        pubsub_msg.a_trajectories = self.a_trajectories

        pubsub_msg.e_Cnt_num_actors = len(self.as_actors_predictions)
        for i in range(pubsub_msg.e_Cnt_num_actors):
            pubsub_msg.as_actors_predictions[i] = self.as_actors_predictions[i].serialize()

        pubsub_msg.e_recipe_description = self.e_recipe_description

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSDataTrajectoryVisualization):
        return cls(pubsubMsg.a_trajectories[:pubsubMsg.e_Cnt_num_trajectories, :pubsubMsg.e_Cnt_num_points_in_trajectory],
                   [PredictionsVisualization.deserialize(pubsubMsg.as_actors_predictions[i])
                    for i in range(pubsubMsg.e_Cnt_num_actors)],
                   pubsubMsg.e_recipe_description)


class TrajectoryVisualizationMsg(PUBSUB_MSG_IMPL):
    def __init__(self, s_Header: Header, s_Data: DataTrajectoryVisualization):
        self.s_Header = s_Header
        self.s_Data = s_Data

    def serialize(self) -> TsSYSTrajectoryVisualization:
        pubsub_msg = TsSYSTrajectoryVisualization()
        pubsub_msg.s_Header = self.s_Header.serialize()
        pubsub_msg.s_Data = self.s_Data.serialize()
        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSTrajectoryVisualization):
        return cls(Header.deserialize(pubsubMsg.s_Header), TrajectoryVisualizationMsg.deserialize(pubsubMsg.s_Data))
