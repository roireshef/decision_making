from interface.Rte_Types.python.sub_structures.TsSYS_DataTrajectoryVisualization import \
    TsSYSDataTrajectoryVisualization
from interface.Rte_Types.python.sub_structures.TsSYS_PredictionsVisualization import \
    TsSYSPredictionsVisualization
from interface.Rte_Types.python.sub_structures.TsSYS_TrajectoryVisualization import \
    TsSYSTrajectoryVisualization
from typing import List

import numpy as np

from decision_making.src.global_constants import PUBSUB_MSG_IMPL
from decision_making.src.messages.scene_common_messages import Header


class PredictionsVisualization(PUBSUB_MSG_IMPL):
    def __init__(self, object_id: int, predictions: np.array):
        """
        The class contains predicted locations for single dynamic object
        :param object_id:
        :param predictions: predicted 2D locations of the object
        """
        self.object_id = object_id
        self.predictions = predictions

    def serialize(self) -> TsSYSPredictionsVisualization:
        pubsub_msg = TsSYSPredictionsVisualization()

        pubsub_msg.e_i_ObjectID = self.object_id
        pubsub_msg.e_Cnt_NumOfPredictions = self.predictions.shape[0]
        pubsub_msg.a_Predictions = self.predictions

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSPredictionsVisualization):
        return cls(pubsubMsg.e_i_ObjectID, pubsubMsg.a_Predictions[:pubsubMsg.e_Cnt_NumOfPredictions])


class DataTrajectoryVisualization(PUBSUB_MSG_IMPL):
    def __init__(self, trajectories: np.ndarray, actors_predictions: List[PredictionsVisualization],
                 recipe_description: str):
        """
        Message that holds debug results of WerlingPlanner to be broadcasted to the visualizer
        :param trajectories: 3D array of trajectories: num_trajectories x trajectory_length x 2
        :param actors_predictions: list of classes of type PredictionsVisualization per dynamic object.
                Each class instance contains predictions for the dynamic object.
        :param recipe_description: String for semantic meaning of action. For example:
                                                            "static action to the left with 50 km/h".
        """
        self.trajectories = trajectories
        self.as_actors_predictions = actors_predictions
        self.recipe_description = recipe_description

    def serialize(self) -> TsSYSDataTrajectoryVisualization:
        pubsub_msg = TsSYSDataTrajectoryVisualization()

        pubsub_msg.e_Cnt_NumOfPointsInTrajectory = self.trajectories.shape[1]
        pubsub_msg.e_Cnt_NumOfTrajectories = self.trajectories.shape[0]
        pubsub_msg.a_Trajectories = self.trajectories

        pubsub_msg.e_Cnt_NumOfActors = len(self.as_actors_predictions)
        for i in range(pubsub_msg.e_Cnt_NumOfActors):
            pubsub_msg.as_ActorsPredictions[i] = self.as_actors_predictions[i].serialize()

        pubsub_msg.a_e_RecipeDescription = self.recipe_description

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSDataTrajectoryVisualization):
        return cls(pubsubMsg.a_Trajectories[:pubsubMsg.e_Cnt_NumOfTrajectories, :pubsubMsg.e_Cnt_NumOfPointsInTrajectory],
                   [PredictionsVisualization.deserialize(pubsubMsg.as_ActorsPredictions[i])
                    for i in range(pubsubMsg.e_Cnt_NumOfActors)],
                   pubsubMsg.a_e_RecipeDescription)


class TrajectoryVisualizationMsg(PUBSUB_MSG_IMPL):
    def __init__(self, header: Header, data: DataTrajectoryVisualization):
        self.header = header
        self.data = data

    def serialize(self) -> TsSYSTrajectoryVisualization:
        pubsub_msg = TsSYSTrajectoryVisualization()
        pubsub_msg.s_Header = self.header.serialize()
        pubsub_msg.s_Data = self.data.serialize()
        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSTrajectoryVisualization):
        return cls(Header.deserialize(pubsubMsg.s_Header), DataTrajectoryVisualization.deserialize(pubsubMsg.s_Data))
