from src.planning.messages.dds_message import DdsMessage


class TrajectoryParameters(DdsMessage):
    def __init__(self, reference_route, target_state, cost_params):
        '''
        The struct used for communicating the behavioral plan to the trajectory planner.
        :param reference_route: of type [(x ,y, theta)] where x, y, theta are floats
        :param target_state: of type (x,y, theta, v) all of which are floats.
        '''
        self._reference_route = reference_route
        self._target_state = target_state
        self._cost_params = cost_params

    def serialize(self):
        pass

    def deserialize(self, message):
        pass
