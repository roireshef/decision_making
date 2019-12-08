class LaneMergeActorState:
    def __init__(self, s_relative_to_ego: float, velocity: float, length: float):
        """
        Actor's state on the main road
        :param s_relative_to_ego: [m] s relative to ego (considering the distance from the actor and from ego to the merge point)
        :param velocity: [m/sec] actor's velocity
        :param length: [m] actor's length
        """
        self.s_relative_to_ego = s_relative_to_ego
        self.velocity = velocity
        self.length = length