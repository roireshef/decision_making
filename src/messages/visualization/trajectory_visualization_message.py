from common_data.interface.Rte_Types.python.sub_structures.TsSYS_DataTrajectoryVisualization import \
    TsSYSDataTrajectoryVisualization
from common_data.interface.Rte_Types.python.sub_structures.TsSYS_PredictionsVisualization import \
    TsSYSPredictionsVisualization
from common_data.interface.Rte_Types.python.sub_structures.TsSYS_TrajectoryVisualization import \
    TsSYSTrajectoryVisualization
from typing import List

import numpy as np

from decision_making.src.global_constants import PUBSUB_MSG_IMPL
from decision_making.src.messages.scene_common_messages import Header


class PredictionsVisualization(PUBSUB_MSG_IMPL):
    def __init__(self, e_i_ObjectID: int, a_Predictions: np.array):
        """
        The class contains predicted locations for single dynamic object
        :param e_i_ObjectID:
        :param a_Predictions: predicted 2D locations of the object
        """
        self.e_i_ObjectID = e_i_ObjectID
        self.a_Predictions = a_Predictions

    def serialize(self) -> TsSYSPredictionsVisualization:
        pubsub_msg = TsSYSPredictionsVisualization()

        pubsub_msg.e_object_id = self.e_i_ObjectID
        pubsub_msg.e_Cnt_num_predictions = self.a_Predictions.shape[0]
        pubsub_msg.a_predictions = self.a_Predictions

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSPredictionsVisualization):
        return cls(pubsubMsg.e_i_ObjectID, pubsubMsg.a_Predictions[:pubsubMsg.e_Cnt_NumOfPredictions])


class DataTrajectoryVisualization(PUBSUB_MSG_IMPL):
    def __init__(self, a_Trajectories: np.ndarray, as_ActorsPredictions: List[PredictionsVisualization],
                 e_recipe_description: str):
        """
        Message that holds debug results of WerlingPlanner to be broadcasted to the visualizer
        :param a_Trajectories: 3D array of trajectories: num_trajectories x trajectory_length x 2
        :param as_ActorsPredictions: list of classes of type PredictionsVisualization per dynamic object.
                Each class instance contains predictions for the dynamic object.
        :param e_recipe_description: String for semantic meaning of action. For example:
                                                            "static action to the left with 50 km/h".
        """
        self.a_Trajectories = a_Trajectories
        self.as_ActorsPredictions = as_ActorsPredictions
        self.e_recipe_description = e_recipe_description

    def serialize(self) -> TsSYSDataTrajectoryVisualization:
        pubsub_msg = TsSYSDataTrajectoryVisualization()

        pubsub_msg.e_Cnt_NumOfPointsInTrajectory = self.a_Trajectories.shape[1]
        pubsub_msg.e_Cnt_NumOfTrajectories = self.a_Trajectories.shape[0]
        pubsub_msg.a_Trajectories = self.a_Trajectories

        pubsub_msg.e_Cnt_NumOfActors = len(self.as_ActorsPredictions)
        for i in range(pubsub_msg.e_Cnt_NumOfActors):
            pubsub_msg.as_ActorsPredictions[i] = self.as_ActorsPredictions[i].serialize()

        pubsub_msg.e_recipe_description = self.e_recipe_description

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSDataTrajectoryVisualization):
        return cls(pubsubMsg.a_Trajectories[:pubsubMsg.e_Cnt_NumOfTrajectories, :pubsubMsg.e_Cnt_NumOfPointsInTrajectory],
                   [PredictionsVisualization.deserialize(pubsubMsg.as_ActorsPredictions[i])
                    for i in range(pubsubMsg.e_Cnt_NumOfActors)],
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
        return cls(Header.deserialize(pubsubMsg.s_Header), DataTrajectoryVisualization.deserialize(pubsubMsg.s_Data))
