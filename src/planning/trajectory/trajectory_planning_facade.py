from common_data.dds.python.Communication.ddspubsub import DdsPubSub
from decision_making.src.infra.dm_module import DmModule
from decision_making.src.messages.trajectory_parameters import TrajectoryParameters
from decision_making.src.messages.trajectory_plan_message import TrajectoryPlanMsg
from decision_making.src.planning.trajectory.trajectory_planner import TrajectoryPlanner
from decision_making.src.planning.trajectory.trajectory_planning_strategy import TrajectoryPlanningStrategy
from decision_making.src.state.enriched_state import EnrichedState
from rte.python.logger.AV_logger import AV_Logger

class TrajectoryPlanningFacade(DmModule):
    def __init__(self, dds: DdsPubSub, logger: AV_Logger, strategy_handlers: dict):
        """
        The trajectory planning facade handles trajectory planning requests and redirects them to the relevant planner
        :param dds: communication layer (DDS) instance
        :param logger: logger
        :param strategy_handlers: a dictionary of trajectory planners as strategy handlers -
        types are {TrajectoryPlanningStrategy: TrajectoryPlanner}
        """
        super().__init__(dds=dds, logger=logger)

        self.__validate_strategy_handlers(strategy_handlers)
        self._strategy_handlers = strategy_handlers

    # TODO: implement
    def _start_impl(self):
        pass

    # TODO: implement
    def _stop_impl(self):
        pass

    # TODO: implement. call plan with the configured strategy
    def _periodic_action_impl(self):
        """
        will execute planning with using the implementation for the desired planning-strategy provided
        :param strategy: desired planning strategy
        :return: no return value. results are published in self.__publish_results()
        """
        state = self.__read_current_state()
        params = self.__read_mission_specs()

        # plan a trajectory according to params (from upper DM level) and most-recent vehicle-state
        trajectory, cost, debug_results = self._strategy_handlers[params.strategy].plan(
            state, params.reference_route, params.target_state, params.cost_params)

        # publish results to the lower DM level
        self.__publish_trajectory(TrajectoryPlanMsg(trajectory=trajectory, reference_route=params.reference_route,
                                                    current_speed=state.ego_state.v_x))

        # publish visualization/debug data
        self.__publish_debug(debug_results)

    # TODO: should also be published to DDS logger
    @staticmethod
    def __validate_strategy_handlers(handlers: dict):
        for elem in TrajectoryPlanningStrategy.__members__.values():
            if not handlers.keys().__contains__(elem):
                raise KeyError('strategy_handlers does not contain a  record for ' + elem)
            if not isinstance(handlers[elem], TrajectoryPlanner):
                raise ValueError('strategy_handlers does not contain a TrajectoryPlanner impl. for ' + elem)

    # TODO: move state into constants file
    def __read_current_state(self) -> EnrichedState:
        input_state = self.dds.get_latest_sample(topic='TrajectoryPlannerSub::StateReader', timeout=1)
        self.logger.info("Recevied state: " + str(input_state))
        return input_state

    # TODO: implement message passing
    def __read_mission_specs(self) -> TrajectoryParameters:
        pass

    # TODO: implement message passing
    def __publish_trajectory(self, results):
        pass

    # TODO: implement message passing
    def __publish_debug(self, debug_data):
        pass
