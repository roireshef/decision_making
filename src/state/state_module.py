import time
from common_data.dds.python.Communication.ddspubsub import DdsPubSub
from decision_making.src.infra.dm_module import DmModule
from decision_making.src.state.enriched_state import *
from rte.python.logger.AV_logger import AV_Logger
from decision_making.src.global_constants import *

class StateModule(DmModule):

    def __init__(self, dds: DdsPubSub, logger):
        super().__init__(dds, logger)

    def start(self):
        self.dds.subscribe(DYNAMIC_OBJECTS_SUBSCRIBE_TOPIC, self.__dynamic_obj_callback)
        self.dds.subscribe(SELF_LOCALIZATION_SUBSCRIBE_TOPIC, self.__self_localization_callback)
        self.dds.subscribe(OCCUPANCY_STATE_SUBSCRIBE_TOPIC, self.__occupancy_state_callback)

    def stop(self):
        self.dds.unsubscribe(DYNAMIC_OBJECTS_SUBSCRIBE_TOPIC)
        self.dds.unsubscribe(SELF_LOCALIZATION_SUBSCRIBE_TOPIC)
        self.dds.unsubscribe(OCCUPANCY_STATE_SUBSCRIBE_TOPIC)

    def periodic_action(self):
        pass

    def __dynamic_obj_callback(self, objects: dict):
        self.logger.info("got dynamic objects %s", objects)

    def __self_localization_callback(self, ego_localization: dict):
        self.logger.debug("got self localization %s", ego_localization)

    def __occupancy_state_callback(self, occupancy: dict):
        self.logger.info("got occupancy status %s", occupancy)

if __name__ == '__main__':
    logger = AV_Logger.get_logger('StateModule')
    dds_object = DdsPubSub("DecisionMakingParticipantLibrary::StateModule",
                           '../../../common_data/dds/generatedFiles/xml/decisionMakingMain.xml')

    state_module = StateModule(dds=dds_object, logger=logger)

    while True:
        # Publish dummy state
        occupancy_state = EnrichedOccupancyState(np.array([0.0]), np.array([0.0]))
        static_objects = [
            EnrichedObjectState(0, 0, 0, 0, 0, 0, EnrichedObjectSize(0, 0, 0), EnrichedRoadLocalization(0, 0, 0, 0, 0),
                                0, 0)]
        dynamic_objects = [EnrichedDynamicObject(0, 0, 0, 0, 0, 0, EnrichedObjectSize(0, 0, 0),
                                                 EnrichedRoadLocalization(0, 0, 0, 0, 0), 0, 0, 0, 0, 0, 0)]
        ego_state = EnrichedEgoState(0, 0, 0, 0, 0, 0, EnrichedObjectSize(0, 0, 0),
                                     EnrichedRoadLocalization(0, 0, 0, 0, 0), 0, 0, 0, 0, 0, 0, 0)
        perceived_road = EnrichedPerceivedRoad(0, [EnrichedLanesStructure(0, 0)], 0)

        enriched_state = EnrichedState(occupancy_state=occupancy_state, static_objects=static_objects,
                                       dynamic_objects=dynamic_objects, ego_state=ego_state,
                                       perceived_road=perceived_road)

        enriched_state_serialized = enriched_state.serialize()
        print (enriched_state_serialized)
        state_module.DDS.publish(topic='StateModulePub::StateWriter', data=enriched_state_serialized)

        time.sleep(1)



