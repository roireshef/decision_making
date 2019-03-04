import traceback
from logging import Logger
from typing import Optional

import numpy as np

from decision_making.src.infra.pubsub import PubSub
from decision_making.src.messages.trajectory_parameters import TrajectoryParams
from decision_making.src.messages.visualization.behavioral_visualization_message import BehavioralVisualizationMsg
from decision_making.src.planning.behavioral.behavioral_planning_facade import BehavioralPlanningFacade
from decision_making.src.planning.types import CartesianPoint2D
from decision_making.test.constants import BP_NEGLIGIBLE_DISPOSITION_LON, BP_NEGLIGIBLE_DISPOSITION_LAT

from decision_making.src.messages.scene_static_message import SceneStatic
from decision_making.src.scene.scene_static_model import SceneStaticModel
from decision_making.src.state.state import State, EgoState
from decision_making.src.utils.map_utils import MapUtils
from decision_making.src.messages.route_plan_message import RoutePlan, DataRoutePlan
from decision_making.src.messages.takeover_message import Takeover, DataTakeover
from decision_making.src.exceptions import MsgDeserializationError, BehavioralPlanningException, StateHasNotArrivedYet ,\
    RepeatedRoadSegments, EgoRoadSegmentNotFound, EgoStationBeyondLaneLength, EgoLaneOccupancyCostIncorrect, \
    RoutePlanningException, MappingException, raises

from decision_making.src.messages.scene_common_messages import Header, Timestamp
from decision_making.src.planning.types import FS_SX
from decision_making.src.global_constants import  DISTANCE_TO_SET_TAKEOVER_FLAG


class BehavioralFacadeMock(BehavioralPlanningFacade):
    """
    Operate according to to policy with an empty dummy behavioral state
    """
    def __init__(self, pubsub: PubSub, logger: Logger, trigger_pos: Optional[CartesianPoint2D],
                 trajectory_params: Optional[TrajectoryParams], visualization_msg: Optional[BehavioralVisualizationMsg]):
        """
        :param pubsub: communication layer (DDS/LCM/...) instance
        :param logger: logger
        :param trigger_pos: the position that triggers the first output, None if there is no need in triggering mechanism
        :param trajectory_params: the trajectory params message to publish periodically
        :param visualization_msg: the visualization message to publish periodically
        """
        super().__init__(pubsub=pubsub, logger=logger, behavioral_planner=None,
                         last_trajectory=None)
        self._trajectory_params = trajectory_params
        self._visualization_msg = visualization_msg

        self._trigger_pos = trigger_pos
        if self._trigger_pos is None:
            self._triggered = True
        else:
            self._triggered = False

    def _periodic_action_impl(self):
        """
        Publishes the received messages initialized in init
        :return: void
        """
        try:
            state = self._get_current_state()
            current_pos = np.array([state.ego_state.x, state.ego_state.y])

            if not self._triggered and np.all(np.abs(current_pos - self._trigger_pos) <
                                              np.array([BP_NEGLIGIBLE_DISPOSITION_LON, BP_NEGLIGIBLE_DISPOSITION_LAT])):
                self._triggered = True

                # NOTE THAT TIMESTAMP IS UPDATED HERE !
                self._trajectory_params.target_time += state.ego_state.timestamp_in_sec

            if self._triggered:
                self._publish_results(self._trajectory_params)
                self._publish_visualization(self._visualization_msg)
            else:
                self.logger.warning("BehavioralPlanningFacade Didn't reach trigger point yet [%s]. "
                                    "Current localization is [%s]" % (self._trigger_pos, current_pos))

        except Exception as e:
            self.logger.error("BehavioralPlanningFacade error %s" % traceback.format_exc())

    @raises(EgoRoadSegmentNotFound, RepeatedRoadSegments, EgoStationBeyondLaneLength, EgoLaneOccupancyCostIncorrect)
    def _mock_takeover_message(self, route_plan_data:DataRoutePlan, ego_state:EgoState, scene_static:SceneStatic) -> Takeover:
        """
        funtion to calculate the takeover message based on the static route plan 
        takeover flag will be set True if all lane segments' end costs for a downstream road segment
        within a threshold distance are 1, i.e., road is blocked. 
        :param route_plan_data: last route plan data 
        :param ego_satte: last state for ego vehicle 
        :scene_static: scene static data to instantiate the sceneStaticModel
        :return: Takeover data
        """
        # additional line to set up MapUtils compared to original _set_takeover_message function
        SceneStaticModel.get_instance().set_scene_static(scene_static)

        ego_lane_segment_id = ego_state.map_state.lane_id

        ego_road_segment_id = MapUtils.get_road_segment_id_from_lane_id(ego_lane_segment_id)

        # find road segment row index in route plan 2-d array 
        route_plan_idx = np.where(route_plan_data.a_i_road_segment_ids == ego_road_segment_id)
        
        if len(route_plan_idx[0]) == 0: # check if ego road segment Id is listed inside route plan data
            raise EgoRoadSegmentNotFound('Route plan does not include data for ego road segment ID {0}'.format(ego_road_segment_id))
        if len(route_plan_idx[0]) > 1 :
            raise RepeatedRoadSegments("Route Plan has repeated data for road segment ID:  \n", ego_road_segment_id)

        row_idx = route_plan_idx[0][0]

        ego_station = ego_state.map_state.lane_fstate[FS_SX]
        
        #find length of the lane segment
        lane = MapUtils.get_lane(ego_lane_segment_id)
        ego_lane_length = lane.e_l_length
        
        dist_to_end = ego_lane_length - ego_station
        
        if  dist_to_end < 0 :
            raise EgoStationBeyondLaneLength("ego station is greater than the lane length for lane segment ID:  \n", ego_lane_segment_id)

        # iterate through all road segments within DISTANCE_TO_SET_TAKEOVER_FLAG 
        for i in range(row_idx, route_plan_data.e_Cnt_num_road_segments):
            
            road_segment_blocked = True

            # check the end cost for all lane segments within a road segment
            for j in range(route_plan_data.a_Cnt_num_lane_segments[i]):
                
                # raise exception if ego lane occupancy cost is 1
                if i == row_idx and route_plan_data.as_route_plan_lane_segments[i][j].e_i_lane_segment_id == ego_lane_segment_id \
                    and  route_plan_data.as_route_plan_lane_segments[i][j].e_cst_lane_occupancy_cost == 1 :
                    raise EgoLaneOccupancyCostIncorrect("Occupancy cost is 1 for ego lane semgnet ID: \n", ego_lane_segment_id)
                                    
                if route_plan_data.as_route_plan_lane_segments[i][j].e_cst_lane_end_cost < 1 :
                    road_segment_blocked = False
                    break

            # continue looking at the next road segments if current road segment is not blocked
            if road_segment_blocked == False :
                
                # find the length of the first lane segment in the next road segment, 
                # assuming that road segment length is similar to its first lane segmnet length
                next_road_segment_lane_id = route_plan_data.as_route_plan_lane_segments[i][0].e_i_lane_segment_id
                lane = MapUtils.get_lane(next_road_segment_lane_id)
                lane_length = lane.e_l_length

                dist_to_end += lane_length

                # check if this road segment lies within the DISTANCE_TO_SET_TAKEOVER_FLAG
                if dist_to_end >= DISTANCE_TO_SET_TAKEOVER_FLAG :
                    break
            else :
                break
        
        if road_segment_blocked == True and dist_to_end < DISTANCE_TO_SET_TAKEOVER_FLAG:
            takeover_flag = True
        else : 
            takeover_flag = False

        takeover_message = Takeover(s_Header=Header(e_Cnt_SeqNum=0, 
                                                    s_Timestamp=Timestamp.from_seconds(ego_state.timestamp_in_sec),
                                                    e_Cnt_version=0),
                                    s_Data=DataTakeover(takeover_flag))

        return takeover_message