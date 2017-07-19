from src.planning.messages.dds_message import DDSMessage


class TrajectoryParameters(DDSMessage):
    def __init__(self, _reference_route: list[(float, float, float)], _target_state: (float, float, float, float),
                 _cost_params: list):
        """
        The struct used for communicating the behavioral plan to the trajectory planner.
        :param _reference_route: of type [(x ,y, theta)] where x, y, theta are floats
        :param _target_state: of type (x,y, theta, v) all of which are floats.
        :param _cost_params: list of parameters for our predefined functions. TODO define this
        """
        self._reference_route = _reference_route
        self._target_state = _target_state
        self._cost_params = _cost_params

    @property
    def reference_route(self): return self._reference_route

    @property
    def target_state(self): return self._target_state

    @property
    def cost_params(self): return self._cost_params
