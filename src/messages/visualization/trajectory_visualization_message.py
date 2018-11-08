from typing import List

import numpy as np

from decision_making.src.global_constants import PUBSUB_MSG_IMPL

from common_data.interface.py.idl_generated_files.dm.sub_structures.LcmNumpyArray import LcmNumpyArray


class PredictionsVisualization(PUBSUB_MSG_IMPL):
    def __init__(self, e_object_id: int, a_predictions: np.array):
        self.e_object_id = e_object_id
        self.a_predictions = a_predictions

    def serialize(self) -> TsSYS_PredictionsVisualization:
        pubsub_msg = TsSYS_PredictionsVisualization()

        pubsub_msg.e_object_id = self.e_object_id

        pubsub_msg.a_predictions = LcmNumpyArray()
        pubsub_msg.a_predictions.num_dimensions = len(self.a_predictions.shape)
        pubsub_msg.a_predictions.shape = list(self.a_predictions.shape)
        pubsub_msg.a_predictions.length = self.a_predictions.size
        pubsub_msg.a_predictions.data = self.a_predictions.ravel()

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYS_PredictionsVisualization):
        predictions_shape = pubsubMsg.a_predictions.shape[:pubsubMsg.a_predictions.num_dimensions]
        predictions_size = np.prod(predictions_shape)
        return cls(pubsubMsg.e_object_id, pubsubMsg.a_predictions.data[:predictions_size].reshape(tuple(predictions_shape)))


class TrajectoryVisualizationMsg(PUBSUB_MSG_IMPL):
    def __init__(self, trajectories: np.ndarray, actors: List[PredictionsVisualization], e_recipe_description: str):
        """
        Message that holds debug results of WerlingPlanner to be broadcasted to the visualizer
        :param trajectories: 3D array of additional trajectories: num_trajectories x trajectory_length x 2
        :param actors: list of predicted objects
        :param e_recipe_description: String for semantic meaning of action
        """
        self.trajectories = trajectories
        self.actors = actors
        self.e_recipe_description = e_recipe_description

    def serialize(self) -> TsSYS_TrajectoryVisualizationMsg:
        pubsub_msg = TsSYS_TrajectoryVisualizationMsg()

        pubsub_msg.trajectories = LcmNumpyArray()
        pubsub_msg.trajectories.num_dimensions = len(self.trajectories.shape)
        pubsub_msg.trajectories.shape = list(self.trajectories.shape)
        pubsub_msg.trajectories.length = self.trajectories.size
        pubsub_msg.trajectories.data = self.trajectories.ravel()

        pubsub_msg.actors = [actor.serialize() for actor in self.actors]
        pubsub_msg.num_actors = len(pubsub_msg.actors)

        pubsub_msg.e_recipe_description = self.e_recipe_description

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYS_TrajectoryVisualizationMsg):
        trajectories_shape = pubsubMsg.trajectories.shape[:pubsubMsg.trajectories.num_dimensions]
        trajectories_size = np.prod(trajectories_shape)
        return cls(pubsubMsg.trajectories.data[:trajectories_size].reshape(tuple(trajectories_shape)),
                   [PredictionsVisualization.deserialize(pubsubMsg.actors[i])
                    for i in range(pubsubMsg.pubsub_msg.num_actors)],
                   pubsubMsg.e_recipe_description)
