from typing import List

import numpy as np

from Rte_Types.sub_structures import TsSYSDataTrajectoryVisualization
from Rte_Types.sub_structures.TsSYS_PredictionsVisualization import TsSYSPredictionsVisualization
from decision_making.src.global_constants import PUBSUB_MSG_IMPL

from common_data.interface.py.idl_generated_files.dm.sub_structures.LcmNonTypedNumpyArray import LcmNonTypedNumpyArray


class PredictionsVisualization(PUBSUB_MSG_IMPL):
    def __init__(self, e_object_id: int, a_predictions: np.array):
        self.e_object_id = e_object_id
        self.a_predictions = a_predictions

    def serialize(self) -> TsSYSPredictionsVisualization:
        pubsub_msg = TsSYSPredictionsVisualization()

        pubsub_msg.e_object_id = self.e_object_id

        pubsub_msg.e_Cnt_num_predictions = self.a_predictions.shape[0]
        pubsub_msg.a_predictions = LcmNonTypedNumpyArray()
        pubsub_msg.a_predictions.num_dimensions = len(self.a_predictions.shape)
        pubsub_msg.a_predictions.shape = list(self.a_predictions.shape)
        pubsub_msg.a_predictions.length = self.a_predictions.size
        pubsub_msg.a_predictions.data = self.a_predictions.ravel()

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSPredictionsVisualization):
        predictions_shape = pubsubMsg.a_predictions.shape[:pubsubMsg.a_predictions.num_dimensions]
        predictions_size = np.prod(predictions_shape)
        return cls(pubsubMsg.e_object_id, pubsubMsg.a_predictions.data[:predictions_size].reshape(tuple(predictions_shape)))


class TrajectoryVisualizationMsg(PUBSUB_MSG_IMPL):
    def __init__(self, a_trajectories: np.ndarray, as_actors_predictions: List[PredictionsVisualization],
                 e_recipe_description: str):
        """
        Message that holds debug results of WerlingPlanner to be broadcasted to the visualizer
        :param a_trajectories: 3D array of additional trajectories: num_trajectories x trajectory_length x 2
        :param as_actors_predictions: list of predicted objects
        :param e_recipe_description: String for semantic meaning of action
        """
        self.a_trajectories = a_trajectories
        self.as_actors_predictions = as_actors_predictions
        self.e_recipe_description = e_recipe_description

    def serialize(self) -> TsSYSDataTrajectoryVisualization:
        pubsub_msg = TsSYSDataTrajectoryVisualization()

        pubsub_msg.e_Cnt_num_trajectories = self.a_trajectories.shape[0]
        pubsub_msg.a_trajectories = LcmNonTypedNumpyArray()
        pubsub_msg.a_trajectories.num_dimensions = len(self.a_trajectories.shape)
        pubsub_msg.a_trajectories.shape = list(self.a_trajectories.shape)
        pubsub_msg.a_trajectories.length = self.a_trajectories.size
        pubsub_msg.a_trajectories.data = self.a_trajectories.ravel()

        pubsub_msg.e_Cnt_num_actors = len(self.as_actors_predictions)
        for i in range(pubsub_msg.e_Cnt_num_actors):
            pubsub_msg.as_actors_predictions[i] = self.as_actors_predictions[i].serialize()

        pubsub_msg.e_recipe_description = self.e_recipe_description

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYSDataTrajectoryVisualization):
        trajectories_shape = pubsubMsg.a_trajectories.shape[:pubsubMsg.a_trajectories.num_dimensions]
        trajectories_size = np.prod(trajectories_shape)
        return cls(pubsubMsg.a_trajectories.data[:trajectories_size].reshape(tuple(trajectories_shape)),
                   [PredictionsVisualization.deserialize(pubsubMsg.as_actors_predictions[i])
                    for i in range(pubsubMsg.e_Cnt_num_actors)],
                   pubsubMsg.e_recipe_description)
