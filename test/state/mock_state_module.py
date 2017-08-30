from common_data.dds.python.Communication.ddspubsub import DdsPubSub
from decision_making.src.global_constants import *
from decision_making.src.map.map_model import MapModel
from decision_making.src.state.state import *
from decision_making.src.state.state_module import StateModule


class StateModuleMock(StateModule):
    """
    Send periodic dummy state message
    """
    def __init__(self, dds: DdsPubSub, logger: Logger):

        occupancy_state = OccupancyState(0, np.array([]), np.array([]))
        dynamic_objects = []
        size = ObjectSize(0, 0, 0)
        map_model = MapModel()
        map_api = MapAPI(map_model)
        road_localization = RoadLocalization(0, 0, 0, 0, 0, 0)
        ego_state = EgoState(0, 0, 0, 0, 0, 0, size, 0, 0, 0, 0, 0, 0, road_localization)

        super().__init__(dds, logger, map_api, occupancy_state, dynamic_objects, ego_state)

    def _periodic_action_impl(self):
        self.__publish_state()

    # TODO: protected instead of private
    def __publish_state(self):

        state = State(self._occupancy_state, self._dynamic_objects, self._ego_state)
        self.dds.publish(STATE_PUBLISH_TOPIC, state.serialize())
