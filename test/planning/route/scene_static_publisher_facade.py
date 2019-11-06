from traceback import format_exc
from logging import Logger
from interface.Rte_Types.python.uc_system import UC_SYSTEM_SCENE_STATIC
from decision_making.src.infra.dm_module import DmModule
from decision_making.src.infra.pubsub import PubSub
from decision_making.src.messages.scene_static_message import SceneStatic
from decision_making.test.planning.route.scene_static_publisher import SceneStaticPublisher


class SceneStaticPublisherFacade(DmModule):
    def __init__(self, pubsub: PubSub, logger: Logger, publisher: SceneStaticPublisher):
        """
        :param pubsub: Middleware
        :param logger: Logger
        :param publisher: Class that creates scene static data
        """
        super().__init__(pubsub=pubsub, logger=logger)
        self.logger.info("Initialized Scene Static Publisher")
        self._publisher = publisher

    def _start_impl(self):
        pass

    def _stop_impl(self):
        pass

    def _periodic_action_impl(self):
        """ The main function that is executed periodically. """
        try:
            # Generate Data and Publish Message
            self._publish_scene_static(self._publisher.generate_data())

        except Exception as e:
            self.logger.critical("SceneStaticPublisher: UNHANDLED EXCEPTION: %s. Trace: %s", e, format_exc())

    def _publish_scene_static(self, scene_static: SceneStatic) -> None:
        """ Publish SCENE_STATIC message """
        self.pubsub.publish(UC_SYSTEM_SCENE_STATIC, scene_static.serialize())
