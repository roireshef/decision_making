import numpy as np
from typing import List

from common_data.interface.py.utils.serialization_utils import SerializationUtils
from decision_making.src.global_constants import PUBSUB_MSG_IMPL
from decision_making.src.planning.types import CartesianPath2D, CartesianExtendedTrajectories
from decision_making.src.state.state import State

from common_data.interface.py.idl_generated_files.Rte_Types.sub_structures.LcmNonTypedSmallNumpyArray import LcmNonTypedSmallNumpyArray
from common_data.interface.py.idl_generated_files.dm import LcmTrajectoryVisualizationMsg


class TrajectoryVisualizationMsg(PUBSUB_MSG_IMPL):
    def __init__(self, reference_route, trajectories, costs, state, predicted_states, plan_time):
        # type: (CartesianPath2D, CartesianExtendedTrajectories, np.ndarray, State, List[State], float) -> None
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
        self.costs = costs
        self.state = state
        self.predicted_states = predicted_states
        self.plan_time = plan_time

    def serialize(self):
        # type: ()->LcmTrajectoryVisualizationMsg
        lcm_msg = LcmTrajectoryVisualizationMsg()

        lcm_msg.reference_route = SerializationUtils.serialize_non_typed_small_array(self.reference_route)
        #lcm_msg.trajectories = SerializationUtils.serialize_non_typed_small_array(self.trajectories)
        lcm_msg.costs = SerializationUtils.serialize_non_typed_small_array(self.costs)

        lcm_msg.state = self.state.serialize()
        lcm_msg.predicted_states = [predicted_state.serialize() for predicted_state in self.predicted_states]
        lcm_msg.num_predicted_states = len(lcm_msg.predicted_states)
        lcm_msg.plan_time = self.plan_time

        return lcm_msg

    @classmethod
    def deserialize(cls, lcmMsg):
        # type: (LcmTrajectoryVisualizationMsg)-> TrajectoryVisualizationMsg
        return cls(SerializationUtils.deserialize_any_array(lcmMsg.reference_route),
                   SerializationUtils.deserialize_any_array(lcmMsg.trajectories),
                   SerializationUtils.deserialize_any_array(lcmMsg.costs),
                   State.deserialize(lcmMsg.state),
                   [State.deserialize(predicted_state) for predicted_state in lcmMsg.predicted_states],
                   lcmMsg.plan_time)

    @property
    def best_trajectory(self):
        return self.trajectories[0]

