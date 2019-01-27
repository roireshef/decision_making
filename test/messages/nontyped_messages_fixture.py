from decision_making.src.messages.dds_nontyped_message import DDSNonTypedMsg
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from typing import List
import numpy as np


class Foo(DDSNonTypedMsg):
    def __init__(self, a, b):
        # type: (float, float) -> None
        self.a = a
        self.b = b


class Voo(DDSNonTypedMsg):
    def __init__(self, x, y):
        # type: (Foo, np.ndarray) -> None
        self.x = x
        self.y = y


class Woo(DDSNonTypedMsg):
    def __init__(self, l):
        # type: (List[Voo]) -> None
        self.l = l


class Moo(DDSNonTypedMsg):
    def __init__(self, strategy):
        # type: (TrajectoryPlanningStrategy) -> None
        self.strategy = strategy
