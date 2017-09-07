import numpy as np

from decision_making.src.messages.dds_typed_message import DDSTypedMsg


class TrajectoryVisualizationMsg(DDSTypedMsg):
    def __init__(self, reference_route, trajectories, costs):
        # type: (np.ndarray, np.ndarray, np.ndarray) -> None
        """
        Message that holds debug results of WerlingPlanner to be broadcasted to the visualizer
        :param reference_route: numpy array the refernce route. please see FrenetMovingFrame.curve documentation
        :param trajectories: a tensor of the best <NUM_ALTERNATIVE_TRAJECTORIES> trajectory points in the vehicle's
        coordinate frame. numpy array of shape [NUM_ALTERNATIVE_TRAJECTORIES, p, 4] where p is the number of points in
        each trajectory and each point consists of [x, y, yaw, velocity]
        :param costs: 1D numpy array of the above trajectories, respectively.
        """
        self.reference_route = reference_route
        self.trajectories = trajectories
        self.costs = costs
