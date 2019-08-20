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
from decision_making.src.state.state import EgoState, State
from decision_making.src.messages.route_plan_message import DataRoutePlan
from decision_making.src.messages.takeover_message import Takeover
from decision_making.src.exceptions import RepeatedRoadSegments, EgoRoadSegmentNotFound, EgoStationBeyondLaneLength, \
    EgoLaneOccupancyCostIncorrect, raises


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

            scene_dynamic = self._get_current_scene_dynamic()

            preprocessed_state = State.create_state_from_scene_dynamic(scene_dynamic, self._last_gff_segment_ids,
                                                                       self.logger)

            state = preprocessed_state.handle_negative_velocities(preprocessed_state, self.logger)

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

        return self._set_takeover_message(route_plan_data, ego_state)