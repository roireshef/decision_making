import numpy as np
from typing import List
from decision_making.src.messages.dds_nontyped_message import DDSNonTypedMsg
from decision_making.src.state.state import State


class TrajectoryVisualizationMsg(DDSNonTypedMsg):
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

    @property
    def best_trajectory(self):
        return self.trajectories[0]
