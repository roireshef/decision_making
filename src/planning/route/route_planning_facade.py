import traceback

from logging import Logger
from common_data.src.communication.pubsub.pubsub import PubSub
from decision_making.src.exceptions import MsgDeserializationError
from decision_making.src.infra.dm_module import DmModule
from decision_making.src.planning.route.route_planner import RoutePlanner

class RoutePlanningFacade(DmModule):
    def __init__(self, pubsub: PubSub, logger: Logger, planner: RoutePlanner):
        """Add comments"""
        super().__init__(pubsub=pubsub, logger=logger)
        self.planner = planner

    def _start_impl(self):
        """Add comments"""
        pass

    def _stop_impl(self):
        """Add comments"""
        pass

    def _periodic_action_impl(self):
        """Add comments"""
        try:
            # Read inputs

            # Plan
            self.__planner.plan()

            # If a takeover is needed, set flag
            if self.__planner.is_takeover_needed():
                pass

            # Write outputs

        except MsgDeserializationError as e:
            self.logger.warning("RoutePlanningFacade: MsgDeserializationError was raised. Skipping planning. " +
                                "Turn on debug logging level for more details. Trace: %s", traceback.format_exc())
            self.logger.debug(str(e))    

        except Exception as e:
            self.logger.critical("RoutePlanningFacade: UNHANDLED EXCEPTION: %s. Trace: %s",
                                 e, traceback.format_exc())

    @property
    def planner(self):
        """Add comments"""
        return self.__planner

    @planner.setter
    def planner(self, planner):
        """Add comments"""
        self.__planner = planner
