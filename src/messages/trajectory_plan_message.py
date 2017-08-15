from decision_making.src.messages.dds_typed_message import DDSTypedMsg
import numpy as np


class TrajectoryPlanMsg(DDSTypedMsg):
    def __init__(self, trajectory: np.ndarray, reference_route: np.ndarray, current_speed: float):
        self.trajectory = trajectory
        self.reference_route = reference_route
        self.current_speed = current_speed
