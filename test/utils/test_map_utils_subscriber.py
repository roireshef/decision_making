import numpy as np
from threading import Lock
from traceback import format_exc
from typing import Any

from common_data.interface.py.pubsub.Rte_Types_pubsub_topics import PubSubMessageTypes
from common_data.src.communication.pubsub.pubsub import PubSub
from common_data.src.communication.pubsub.pubsub_factory import create_pubsub

from common_data.interface.py.idl_generated_files.Rte_Types.TsSYS_SceneStatic import TsSYSSceneStatic
from common_data.interface.py.pubsub.Rte_Types_pubsub_topics import SCENE_STATIC
from common_data.lcm.config import pubsub_topics

from decision_making.src.messages.scene_static_message import SceneStatic
from decision_making.src.mapping.scene_model import SceneModel
from decision_making.src.utils.map_utils import MapUtils


class SceneSubscriber():
    def __init__(self):
        self.pubsub = create_pubsub(PubSubMessageTypes)
        self._scene_static_lock = Lock()
        self._scene_static = None
        self.running = False

    def start_sub(self) -> None:
        self.pubsub.subscribe(SCENE_STATIC, self._scene_static_callback)
        self.running = True
        print("INFO: Scene subscription active")

    def stop_sub(self) -> None:
        self.pubsub.unsubscribe(SCENE_STATIC)    
        self.running = False
        print("INFO: Scene subscription inactive")

    def _scene_static_callback(self, scene_static: TsSYSSceneStatic, args: Any):
        try:
            with self._scene_static_lock:
                self._scene_static = SceneStatic.deserialize(scene_static)
                self._add_to_scene_model()

        except Exception as e:
            print("ERROR: StateModule._scene_dynamic_callback failed due to %s", format_exc())

    def _add_to_scene_model(self):
        SceneModel.get_instance().add_scene_static(self._scene_static)


def main():
    scene_sub = SceneSubscriber()
    if scene_sub: 
        scene_sub.start_sub()

    try:
        while scene_sub.running:
            
            #######################################################
            # Insert tests here
            #######################################################             
            pass
    except KeyboardInterrupt:
        print("INFO: Ctrl+C pressed")
    finally:
        scene_sub.stop_sub()

if __name__ == '__main__':
    main()
