from enum import Enum
from planning.trajectory.trajectory_planner import TrajectoryPlanner


class TrajectoryPlanningFacade:
    def __init__(self, strategy_handlers: dict):
        """
        The trajectory planning facade handles trajectory planning requests and redirects them to the relevant planner
        :param strategy_handlers: a dictionary of {strategy: planner} where strategy is taken from
        TrajectoryPlanningStrategy's keys and planner is a TrajectoryPlanner impl.
        """
        self.__validate_strategy_handlers(strategy_handlers)
        self.strategy_handlers = strategy_handlers

    def plan(self, strategy: TrajectoryPlanningStrategy):
        state, ref_route, goal, cost_params = self.__get_mission_specs()
        results = self.strategy_handlers[strategy.name].plan(state, ref_route, goal, cost_params)
        self.__publish_results(results)

    @staticmethod
    def __validate_strategy_handlers(strategy_handlers):
        for name, _ in TrajectoryPlanningStrategy.__members__.items():
            assert strategy_handlers.keys().__contains__(name), 'strategy_handlers does not contain a ' \
                                                                     'record for ' + name
            assert isinstance(strategy_handlers[name], TrajectoryPlanner), 'strategy_handlers does not contain a ' \
                                                                           'TrajectoryPlanner impl. for ' + name

    # TODO: implement message passing
    def __get_current_state(self):
        pass

    def __get_mission_specs(self):
        pass

    def __publish_results(self):
        pass


class TrajectoryPlanningStrategy(Enum):
    HIGHWAY = 'highway'
    TRAFFIC_JAM = 'traffic_jam'
    PARKING = 'parking'
