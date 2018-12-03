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

from decision_making.src.messages.scene_static_message import SceneStatic, NominalPathPoint
from decision_making.src.mapping.scene_model import SceneModel
# from decision_making.src.utils.map_utils import MapUtils


class SceneSubscriber():
    def __init__(self, limit_for_num_msgs=False, msg_limit=None):
        self.pubsub = create_pubsub(PubSubMessageTypes)
        self._scene_static_lock = Lock()
        self._scene_static = None
        self._limit_for_num_msgs = limit_for_num_msgs
        self._msg_limit = msg_limit
        self.running = False

    def start_sub(self) -> None:
        self.pubsub.subscribe(SCENE_STATIC, self._scene_static_callback)
        self.running = True
        self._num_msgs_rcvd = 0
        print("INFO: Scene subscription active")

    def stop_sub(self) -> None:
        self.pubsub.unsubscribe(SCENE_STATIC)    
        self.running = False
        print("INFO: Scene subscription inactive")

    def _scene_static_callback(self, scene_static: TsSYSSceneStatic, args: Any):
        try:
            with self._scene_static_lock:
                if self._limit_for_num_msgs:
                    if self._num_msgs_rcvd < self._msg_limit:
                        self._scene_static = SceneStatic.deserialize(scene_static)
                        self._add_to_scene_model()
                else:
                    self._scene_static = SceneStatic.deserialize(scene_static)
                    self._add_to_scene_model()
                
                self._num_msgs_rcvd += 1

        except Exception as e:
            print("ERROR: StateModule._scene_dynamic_callback failed due to %s", format_exc())

    def _add_to_scene_model(self):
        SceneModel.get_instance().add_scene_static(self._scene_static)

    def print_static_scene_message(self):
        print("SCENE MESSAGE #", self._num_msgs_rcvd)
        print("rear percep horizon=", self._scene_static.s_Data.e_l_perception_horizon_rear, ", front=", self._scene_static.s_Data.e_l_perception_horizon_front)
        print("num road segs = ", self._scene_static.s_Data.e_Cnt_num_road_segments, ", road segs:")
        roadseg_idx = 0
        for roadseg in self._scene_static.s_Data.as_scene_road_segment:
            print("\troad seg #", roadseg_idx, ", ID=", roadseg.e_Cnt_road_segment_id, ", road ID=", roadseg.e_Cnt_road_id)
            lane_seg_ids = [(str(laneid) + " ") for laneid in roadseg.a_Cnt_lane_segment_id]
            print("\tnum lane segs=", roadseg.e_Cnt_lane_segment_id_count, ": ", lane_seg_ids)
            upstream_road_ids = [roadid for roadid in roadseg.a_Cnt_upstream_road_segment_id]
            print("\tnum upstream road segs=", roadseg.e_Cnt_upstream_segment_count, ": ", upstream_road_ids)
            downstream_road_ids = [roadid for roadid in roadseg.a_Cnt_downstream_road_segment_id]
            print("\tnum upstream road segs=", roadseg.e_Cnt_downstream_segment_count, ": ", downstream_road_ids)
            roadseg_idx += 1

        laneseg_idx = 0
        print("num lane segs = ", self._scene_static.s_Data.e_Cnt_num_lane_segments, ", lane segs:")
        for laneseg in self._scene_static.s_Data.as_scene_lane_segment:
            print("\tlane seg #", laneseg_idx, ", ID=", laneseg.e_i_lane_segment_id, ", roadseg ID=", laneseg.e_i_road_segment_id)
            radj_lane_seg_ids = [laneid.e_Cnt_lane_segment_id for laneid in laneseg.as_right_adjacent_lanes]
            print("\tnum right adj lane segs=", laneseg.e_Cnt_right_adjacent_lane_count, ": ", radj_lane_seg_ids)
            ladj_lane_seg_ids = [laneid.e_Cnt_lane_segment_id for laneid in laneseg.as_left_adjacent_lanes]
            print("\tnum left adj lane segs=", laneseg.e_Cnt_left_adjacent_lane_count, ": ", ladj_lane_seg_ids)
            upstream_lane_ids = [laneid.e_Cnt_lane_segment_id for laneid in laneseg.as_upstream_lanes]
            print("\tnum upstream lane segs=", laneseg.e_Cnt_upstream_lane_count, ": ", upstream_lane_ids)
            downstream_lane_ids = [laneid.e_Cnt_lane_segment_id for laneid in laneseg.as_downstream_lanes]
            print("\tnum upstream lane segs=", laneseg.e_Cnt_downstream_lane_count, ": ", downstream_lane_ids)
            print("\tnum nominal pts=", laneseg.e_Cnt_nominal_path_point_count, "nom path pts:")
            nom_pt_idx = 0
            for pt in laneseg.a_nominal_path_points:
                print("\t\tPT #", nom_pt_idx, " at E=", pt[NominalPathPoint.CeSYS_NominalPathPoint_e_l_EastX.value], " N=", pt[NominalPathPoint.CeSYS_NominalPathPoint_e_l_NorthY.value])
                print("\t\tphi heading=", pt[NominalPathPoint.CeSYS_NominalPathPoint_e_phi_heading.value], ", s=", pt[NominalPathPoint.CeSYS_NominalPathPoint_e_l_s.value])
                print("\t\tleft offset=", pt[NominalPathPoint.CeSYS_NominalPathPoint_e_l_left_offset.value], ", right offset=", pt[NominalPathPoint.CeSYS_NominalPathPoint_e_l_right_offset.value])
                nom_pt_idx += 1
            laneseg_idx += 1


def main():
    # scene_sub = SceneSubscriber()
    # Use above to subscribe to every message until Ctrl+C
    # Use below if want to limit subscriber to one scene static message
    scene_sub = SceneSubscriber(True, 1)
    
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
