from decision_making.src.messages.dds_nontyped_message import DDSNonTypedMsg
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy


class Foo(DDSNonTypedMsg):
    def __init__(self, a, b):
        self.a = a
        self.b = b


class Voo(DDSNonTypedMsg):
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Woo(DDSNonTypedMsg):
    def __init__(self, l):
        self.l = l


class Moo(DDSNonTypedMsg):
    def __init__(self, strategy: TrajectoryPlanningStrategy):
        self.strategy = strategy
