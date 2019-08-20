from logging import Logger

from common_data.interface.Rte_Types.python.uc_system import UC_SYSTEM_SCENE_DYNAMIC
from decision_making.src.infra.pubsub import PubSub
from decision_making.src.messages.scene_dynamic_message import SceneDynamic
from decision_making.src.infra.dm_module import DmModule


class SceneDynamicMock(DmModule):
    def __init__(self, pubsub: PubSub, logger: Logger, scene_dynamic: SceneDynamic):
        super().__init__(pubsub, logger, None)
        self._scene_dynamic = scene_dynamic

    def _periodic_action_impl(self) -> None:
        self.pubsub.publish(UC_SYSTEM_SCENE_DYNAMIC, self._scene_dynamic.serialize())

    def _start_impl(self):
        pass

    def _stop_impl(self):
        pass