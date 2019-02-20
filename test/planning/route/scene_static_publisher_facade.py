from traceback import format_exc
from logging import Logger

from common_data.interface.Rte_Types.python import Rte_Types_pubsub as pubsub_topics

from decision_making.src.infra.dm_module import DmModule
from decision_making.src.infra.pubsub import PubSub
from decision_making.src.messages.scene_static_message import SceneStatic
from decision_making.test.planning.route.scene_static_publisher import SceneStaticPublisher

class SceneStaticPublisherFacade(DmModule):
    """
    TODO: Add description

    Args:
        pubsub: TODO: Add description
        logger: TODO: Add description
    """
    def __init__(self, pubsub: PubSub, logger: Logger, publisher: SceneStaticPublisher):
        super().__init__(pubsub=pubsub, logger=logger)
        self.logger.info("Initialized Scene Static Publisher")
        self._publisher = publisher

    def _start_impl(self):
        """ TODO: Add description """
        pass

    def _stop_impl(self):
        """ TODO: Add description """
        pass

    def _periodic_action_impl(self):
        """ TODO: Add description """
        try:
            # Hard-coded scene variables
            num_road_segments = 2
            road_segment_ids = [1, 2]

            num_lane_segments = 4
            lane_segment_ids = [[101, 102],
                                [201, 202]]

            navigation_plan = [1, 2]

            # Generate Data and Publish Message
            self._publish_scene_static(self._publisher.generate_data(num_road_segments=num_road_segments,
                                                                     road_segment_ids=road_segment_ids,
                                                                     num_lane_segments=num_lane_segments,
                                                                     lane_segment_ids=lane_segment_ids,
                                                                     navigation_plan=navigation_plan))

        except Exception as e:
            self.logger.critical("SceneStaticPublisher: UNHANDLED EXCEPTION: %s. Trace: %s",
                                 e, format_exc())

    def _publish_scene_static(self, scene_static: SceneStatic) -> None:
        """ Publish SCENE_STATIC message """
        self.pubsub.publish(pubsub_topics.PubSubMessageTypes["UC_SYSTEM_SCENE_STATIC"], scene_static.serialize())
