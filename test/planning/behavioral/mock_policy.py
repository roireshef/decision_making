import numpy as np
from decision_making.src.messages.trajectory_parameters import TrajectoryParams, TrajectoryCostParams, \
    SigmoidFunctionParams
from decision_making.src.messages.visualization.behavioral_visualization_message import BehavioralVisualizationMsg
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState
from decision_making.src.planning.behavioral.policy import Policy
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy


class PolicyMock(Policy):
    """
    Dummy policy
    """
    def __init__(self):
        policy_params = dict()
        super().__init__(policy_params=policy_params)

    def plan(self, behavioral_state: BehavioralState):
        ref_route = np.array(
            [[1.0, -2.0, 0.0], [2.0, -2.0, 0.0], [3.0, -2.0, 0.0], [4.0, -2.0, 0.0], [5.0, -2.0, 0.0], [6.0, -2.0, 0.0],
             [7.0, -2.0, 0.0], [8.0, -2.0, 0.0], [9.0, -2.0, 0.0], [10.0, -2.0, 0.0], [11.0, -2.0, 0.0],
             [12.0, -2.0, 0.0], [13.0, -2.0, 0.0], [14.0, -2.0, 0.0], [15.0, -2.0, 0.0], [16.0, -2.0, 0.0]])
        target_state = np.array([16.0, -2.0, 0.0])
        mock_sigmoid = SigmoidFunctionParams(0.0, 0.0, 0.0)
        trajectory_cost_params = TrajectoryCostParams(mock_sigmoid, mock_sigmoid, mock_sigmoid, mock_sigmoid,
                                                      mock_sigmoid, mock_sigmoid, mock_sigmoid, 0.0,
                                                      np.array([1.0]), np.array([1.0]))
        trajectory_parameters = TrajectoryParams(reference_route=ref_route, target_state=target_state,
                                                 cost_params=trajectory_cost_params, time=0,
                                                 strategy=TrajectoryPlanningStrategy.HIGHWAY)

        visualization_message = BehavioralVisualizationMsg(reference_route=ref_route)
        return trajectory_parameters, visualization_message
