from typing import List

import numpy as np

from Rte_Types import TsSYSTrajectoryVisualization
from Rte_Types.sub_structures import TsSYSDataTrajectoryVisualization, TsSYSTimestamp, TsSYSHeader
from Rte_Types.sub_structures.TsSYS_PredictionsVisualization import TsSYSPredictionsVisualization
from decision_making.src.global_constants import PUBSUB_MSG_IMPL


class Timestamp(PUBSUB_MSG_IMPL):
    e_Cnt_Secs = int
    # TODO: why fractions are int?
    e_Cnt_FractionSecs = int

    def __init__(self, e_Cnt_Secs, e_Cnt_FractionSecs):
        # type: (int, int)->None
        """
        A data class that corresponds to a parametrization of a sigmoid function
        :param e_Cnt_Secs: Seconds since 1 January 1900
        :param e_Cnt_FractionSecs: Fractional seconds
        """
        self.e_Cnt_Secs = e_Cnt_Secs
        self.e_Cnt_FractionSecs = e_Cnt_FractionSecs

    def serialize(self):
        # type: () -> TsSYSTimestamp
        pubsub_msg = TsSYSTimestamp()

        pubsub_msg.e_Cnt_Secs = self.e_Cnt_Secs
        pubsub_msg.e_Cnt_FractionSecs = self.e_Cnt_FractionSecs

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg):
        # type: (TsSYSTimestamp)->Timestamp
        return cls(pubsubMsg.e_Cnt_Secs, pubsubMsg.e_Cnt_FractionSecs)


class Header(PUBSUB_MSG_IMPL):
    e_Cnt_SeqNum = int
    s_Timestamp = Timestamp
    e_Cnt_version = int

    def __init__(self, e_Cnt_SeqNum, s_Timestamp, e_Cnt_version):
        # type: (int, Timestamp, int)->None
        """
        Header Information is controlled by Middleware
        :param e_Cnt_SeqNum: Starts from 0 and increments at every update of this data structure
        :param s_Timestamp: Timestamp in secs and nano seconds when the data was published
        :param e_Cnt_version: Version of the topic/service used to identify interface compatability
        :return:
        """
        self.e_Cnt_SeqNum = e_Cnt_SeqNum
        self.s_Timestamp = s_Timestamp
        self.e_Cnt_version = e_Cnt_version

    def serialize(self):
        # type: () -> TsSYSHeader
        pubsub_msg = TsSYSHeader()

        pubsub_msg.e_Cnt_SeqNum = self.e_Cnt_SeqNum
        pubsub_msg.s_Timestamp = self.s_Timestamp.serialize()
        pubsub_msg.e_Cnt_version = self.e_Cnt_version

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg):
        # type: (TsSYSHeader)->Header
        return cls(pubsubMsg.e_Cnt_SeqNum, Timestamp.deserialize(pubsubMsg.s_Timestamp), pubsubMsg.e_Cnt_version)


class PredictionsVisualization(PUBSUB_MSG_IMPL):
    def __init__(self, e_object_id: int, a_predictions: np.array):
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
        :param a_trajectories: 3D array of additional trajectories: num_trajectories x trajectory_length x 2
        :param as_actors_predictions: list of predicted objects
        :param e_recipe_description: String for semantic meaning of action
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
        return cls(pubsubMsg.a_trajectories[:pubsubMsg.e_Cnt_num_trajectories][:pubsubMsg.e_Cnt_num_points_in_trajectory],
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
