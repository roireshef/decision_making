import numpy as np
from typing import List
from decision_making.src.state.state import State

from common_data.lcm.generatedFiles.gm_lcm import LcmNonTypedNumpyArray
from common_data.lcm.generatedFiles.gm_lcm import LcmTrajectoryVisualizationMsg

class TrajectoryVisualizationMsg:
    def __init__(self, reference_route, trajectories, costs, state, predicted_states, plan_time):
        # type: (np.ndarray, np.ndarray, np.ndarray, State, List[State], float) -> None
        """
        Message that holds debug results of WerlingPlanner to be broadcasted to the visualizer
        :param reference_route: numpy array the refernce route. please see FrenetMovingFrame.curve documentation
        :param trajectories: a tensor of the best <NUM_ALTERNATIVE_TRAJECTORIES> trajectory points in the vehicle's
        coordinate frame. numpy array of shape [NUM_ALTERNATIVE_TRAJECTORIES, p, 4] where p is the number of points in
        each trajectory and each point consists of [x, y, yaw, velocity]
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

    def to_lcm(self) -> LcmTrajectoryVisualizationMsg:
        lcm_msg = LcmTrajectoryVisualizationMsg()

        lcm_msg.reference_route = LcmNonTypedNumpyArray()
        lcm_msg.reference_route.num_dimensions = len(self.reference_route.shape)
        lcm_msg.reference_route.shape = list(self.reference_route.shape)
        lcm_msg.reference_route.length = self.reference_route.size
        lcm_msg.reference_route.data = self.reference_route.flat.__array__().tolist()

        lcm_msg.trajectories = LcmNonTypedNumpyArray()
        lcm_msg.trajectories.num_dimensions = len(self.trajectories.shape)
        lcm_msg.trajectories.shape = list(self.trajectories.shape)
        lcm_msg.trajectories.length = self.trajectories.size
        lcm_msg.trajectories.data = self.trajectories.flat.__array__().tolist()

        lcm_msg.costs = LcmNonTypedNumpyArray()
        lcm_msg.costs.num_dimensions = len(self.costs.shape)
        lcm_msg.costs.shape = list(self.costs.shape)
        lcm_msg.costs.length = self.costs.size
        lcm_msg.costs.data = self.costs.flat.__array__().tolist()

        lcm_msg.state = self.state.to_lcm()
        lcm_msg.predicted_states = [predicted_state.to_lcm() for predicted_state in self.predicted_states]
        lcm_msg.num_predicted_states = len(lcm_msg.predicted_states)
        lcm_msg.plan_time = self.plan_time

        return lcm_msg

    @classmethod
    def from_lcm(cls, lcmMsg: LcmTrajectoryVisualizationMsg):
        return cls(np.ndarray(shape = tuple(lcmMsg.reference_route.shape)
                            , buffer = np.array(lcmMsg.reference_route.data)
                            , dtype = float)
                 , np.ndarray(shape = tuple(lcmMsg.trajectories.shape)
                            , buffer = np.array(lcmMsg.trajectories.data)
                            , dtype = float)
                 , np.ndarray(shape = tuple(lcmMsg.costs.shape)
                            , buffer = np.array(lcmMsg.costs.data)
                            , dtype = float)
                 , State.from_lcm(lcmMsg.state)
                 , [State.from_lcm(predicted_state) for predicted_state in lcmMsg.predicted_states]
                 , lcmMsg.plan_time)

    @property
    def best_trajectory(self):
        return self.trajectories[0]

