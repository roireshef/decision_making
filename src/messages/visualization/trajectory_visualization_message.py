from typing import List

import numpy as np

from decision_making.src.global_constants import PUBSUB_MSG_IMPL
from decision_making.src.planning.types import CartesianPath2D, CartesianPoint2D, CartesianExtendedState

from common_data.interface.py.idl_generated_files.dm.sub_structures.LcmNumpyArray import LcmNumpyArray
from common_data.interface.py.idl_generated_files.dm import TsSYS_TrajectoryVisualizationMsg


class GoalVisualization(PUBSUB_MSG_IMPL):
    def __init__(self, e_plan_horizon: float, s_cartesian_loc: CartesianPoint2D,
                 e_target_velocity: float, e_recipe_description: str):
        """
        Visualization message of Goal. The goal is output of BP and input of TP. It's a point specified by the chosen
        action, that the output trajectory attempts to achieve.
        :param e_plan_horizon: longitudinal planning time for the goal reaching
        :param s_cartesian_loc: 2D cartesian location of the goal
        :param e_target_velocity: target velocity on goal
        :param e_recipe_description: String for semantic meaning of action. For example:
                    "static action right lane 10 m/s", "follow front left vehicle"
        """
        self.e_plan_horizon = e_plan_horizon
        self.s_cartesian_loc = s_cartesian_loc
        self.e_target_velocity = e_target_velocity
        self.e_recipe_description = e_recipe_description

    def serialize(self) -> TsSYS_GoalVisualization:
        pubsub_msg = TsSYS_GoalVisualization()

        pubsub_msg.e_plan_horizon = self.e_plan_horizon
        pubsub_msg.s_cartesian_loc = self.s_cartesian_loc
        pubsub_msg.e_target_velocity = self.e_target_velocity
        pubsub_msg.e_recipe_description = self.e_recipe_description

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYS_GoalVisualization):
        return cls(pubsubMsg.e_plan_horizon, pubsubMsg.s_cartesian_loc, pubsubMsg.e_target_velocity,
                   pubsubMsg.e_recipe_description)


class ObjectVisualization(PUBSUB_MSG_IMPL):
    def __init__(self, e_object_id: int, a_box_dimensions: np.array, a_predictions: np.array):
        self.e_object_id = e_object_id
        self.a_box_dimensions = a_box_dimensions
        self.a_predictions = a_predictions

    def serialize(self) -> TsSYS_ObjectVisualization:
        pubsub_msg = TsSYS_ObjectVisualization()

        pubsub_msg.e_object_id = self.e_object_id
        pubsub_msg.a_box_dimensions = self.a_box_dimensions

        pubsub_msg.a_predictions = LcmNumpyArray()
        pubsub_msg.a_predictions.num_dimensions = len(self.a_predictions.shape)
        pubsub_msg.a_predictions.shape = list(self.a_predictions.shape)
        pubsub_msg.a_predictions.length = self.a_predictions.size
        pubsub_msg.a_predictions.data = self.a_predictions.ravel()

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYS_ObjectVisualization):
        return cls(pubsubMsg.e_object_id,
                   pubsubMsg.a_box_dimensions,
                   pubsubMsg.a_predictions.data.reshape(
                       tuple(pubsubMsg.a_predictions.shape[:pubsubMsg.a_predictions.num_dimensions])))


class TrajectoryVisualizationMsg(PUBSUB_MSG_IMPL):
    def __init__(self, host_pose: CartesianExtendedState, goal: GoalVisualization, reference_route: CartesianPath2D,
                 trajectories: np.ndarray, actors: List[ObjectVisualization]):
        """
        Message that holds debug results of WerlingPlanner to be broadcasted to the visualizer
        :param host_pose: host cartesian state (array of size 6)
        :param goal: the target pose of the action
        :param reference_route: the target lane center route
        :param trajectories: 3D array of additional trajectories: num_trajectories x trajectory_length x 2
        :param actors: list of predicted objects
        """
        self.host_pose = host_pose
        self.goal = goal
        self.reference_route = reference_route
        self.trajectories = trajectories
        self.actors = actors

    def serialize(self) -> TsSYS_TrajectoryVisualizationMsg:
        pubsub_msg = TsSYS_TrajectoryVisualizationMsg()

        pubsub_msg.host_pose = self.host_pose
        pubsub_msg.goal = self.goal.serialize()

        pubsub_msg.reference_route = LcmNumpyArray()
        pubsub_msg.reference_route.num_dimensions = len(self.reference_route.shape)
        pubsub_msg.reference_route.shape = list(self.reference_route.shape)
        pubsub_msg.reference_route.length = self.reference_route.size
        pubsub_msg.reference_route.data = self.reference_route.ravel()

        pubsub_msg.trajectories = LcmNumpyArray()
        pubsub_msg.trajectories.num_dimensions = len(self.trajectories.shape)
        pubsub_msg.trajectories.shape = list(self.trajectories.shape)
        pubsub_msg.trajectories.length = self.trajectories.size
        pubsub_msg.trajectories.data = self.trajectories.ravel()

        pubsub_msg.actors = [actor.serialize() for actor in self.actors]
        pubsub_msg.num_actors = len(pubsub_msg.actors)

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg: TsSYS_TrajectoryVisualizationMsg):
        return cls(pubsubMsg.host_pose,
                   pubsubMsg.goal.deserialize(),
                   pubsubMsg.reference_route.data.reshape(
                       tuple(pubsubMsg.reference_route.shape[:pubsubMsg.reference_route.num_dimensions])),
                   pubsubMsg.trajectories.data.reshape(
                       tuple(pubsubMsg.trajectories.shape[:pubsubMsg.trajectories.num_dimensions])),
                   [ObjectVisualization.deserialize(pubsubMsg.actors[i])
                    for i in range(pubsubMsg.pubsub_msg.num_actors)])
