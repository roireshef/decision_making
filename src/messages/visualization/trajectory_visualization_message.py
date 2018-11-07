import numpy as np

from decision_making.src.global_constants import PUBSUB_MSG_IMPL
from decision_making.src.planning.types import CartesianPath2D, CartesianPoint2D, FrenetState2D
from decision_making.src.state.state import State

from common_data.interface.py.idl_generated_files.dm.sub_structures.LcmNonTypedSmallNumpyArray import LcmNonTypedSmallNumpyArray
from common_data.interface.py.idl_generated_files.dm import LcmTrajectoryVisualizationMsg


class GoalVisualization(PUBSUB_MSG_IMPL):
    def __init__(self, s_timestamp: float, s_cartesian_loc: CartesianPoint2D, e_lane_segment_id: int,
                 s_lane_frenet_state: FrenetState2D, e_recipe_description: str):
        """
        Visualization message of Goal. The goal is output of BP and input of TP. It's a point specified by the chosen
        action, that the output trajectory attempts to achieve.
        :param s_timestamp: timestamp for the goal reaching
        :param s_cartesian_loc: 2D cartesian point of the goal
        :param e_lane_segment_id: lane ID of the goal
        :param s_lane_frenet_state: frenet state of the goal w.r.t. the lane
        :param e_recipe_description: String for semantic meaning of action. For example:
                    "static action right lane 10 m/s", "follow front left vehicle"
        """
        self.s_timestamp = s_timestamp
        self.s_cartesian_loc = s_cartesian_loc
        self.e_lane_segment_id = e_lane_segment_id
        self.s_lane_frenet_state = s_lane_frenet_state
        self.e_recipe_description = e_recipe_description

    def serialize(self) -> TsSYS_VisualizationGoal:
        pubsub_msg = TsSYSObjectHypothesis()

        pubsub_msg.s_timestamp = self.s_timestamp
        pubsub_msg.s_cartesian_loc = self.s_cartesian_loc
        pubsub_msg.e_lane_segment_id = self.e_lane_segment_id
        pubsub_msg.s_lane_frenet_state = self.s_lane_frenet_state
        pubsub_msg.e_recipe_description = self.e_recipe_description

        return pubsub_msg


class TrajectoryVisualizationMsg(PUBSUB_MSG_IMPL):
    def __init__(self, reference_route: CartesianPath2D, trajectories: np.ndarray, scene_dynamic: DataSceneDynamic,
                 predicted_states: np.ndarray, plan_time: float):
        """
        Message that holds debug results of WerlingPlanner to be broadcasted to the visualizer
        :param reference_route: of type CartesianPath2D
        :param trajectories: a tensor of the best <NUM_ALTERNATIVE_TRAJECTORIES> trajectory points in the vehicle's
        coordinate frame. of type CartesianExtendedTrajectories
        :param costs: 1D numpy array of the above trajectories, respectively.
        :param state: the current state
        :param predicted_states: a list of predicted states (uniform prediction times in ascending order).
        Only the predictions of the dynamic objects are used.
        :param plan_time: the time given to the trajectory planner for trajectory generation
        """
        self.reference_route = reference_route
        self.trajectories = trajectories
        self.scene_dynamic = scene_dynamic
        self.predicted_states = predicted_states
        self.plan_time = plan_time

    def serialize(self) -> TsSYS_TrajectoryVisualizationMsg:
        pubsub_msg = LcmTrajectoryVisualizationMsg()

        pubsub_msg.reference_route = LcmNonTypedSmallNumpyArray()
        pubsub_msg.reference_route.num_dimensions = len(self.reference_route.shape)
        pubsub_msg.reference_route.shape = list(self.reference_route.shape)
        pubsub_msg.reference_route.length = self.reference_route.size
        pubsub_msg.reference_route.data = self.reference_route.flat.__array__().tolist()

        pubsub_msg.trajectories = LcmNonTypedSmallNumpyArray()
        pubsub_msg.trajectories.num_dimensions = len(self.trajectories.shape)
        pubsub_msg.trajectories.shape = list(self.trajectories.shape)
        pubsub_msg.trajectories.length = self.trajectories.size
        pubsub_msg.trajectories.data = self.trajectories.flat.__array__().tolist()

        pubsub_msg.state = self.state.serialize()
        pubsub_msg.predicted_states = [predicted_state.serialize() for predicted_state in self.predicted_states]
        pubsub_msg.num_predicted_states = len(pubsub_msg.predicted_states)
        pubsub_msg.plan_time = self.plan_time

        return pubsub_msg

    @classmethod
    def deserialize(cls, pubsubMsg):
        # type: (LcmTrajectoryVisualizationMsg)-> TrajectoryVisualizationMsg
        return cls(np.ndarray(shape = tuple(pubsubMsg.reference_route.shape[:pubsubMsg.reference_route.num_dimensions])
                            , buffer = np.array(pubsubMsg.reference_route.data)
                            , dtype = float)
                 , np.ndarray(shape = tuple(pubsubMsg.trajectories.shape[:pubsubMsg.trajectories.num_dimensions])
                            , buffer = np.array(pubsubMsg.trajectories.data)
                            , dtype = float)
                 , np.ndarray(shape = tuple(pubsubMsg.costs.shape[:pubsubMsg.costs.num_dimensions])
                            , buffer = np.array(pubsubMsg.costs.data)
                            , dtype = float)
                 , State.deserialize(pubsubMsg.state)
                 , [State.deserialize(predicted_state) for predicted_state in pubsubMsg.predicted_states]
                 , pubsubMsg.plan_time)

    @property
    def best_trajectory(self):
        return self.trajectories[0]

