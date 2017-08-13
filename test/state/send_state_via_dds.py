import time

from common_data.dds.python.Communication.ddspubsub import DdsPubSub
from decision_making.src.state.enriched_state import *
from decision_making.src.state.state_module import StateModule
from rte.python.logger.AV_logger import AV_Logger

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
    perceived_road = EnrichedPerceivedRoad(0, [EnrichedLanesStructure(np.array([0.0]), np.array([0.0]))], 0)

    enriched_state = EnrichedState(occupancy_state=occupancy_state, static_objects=static_objects,
                                   dynamic_objects=dynamic_objects, ego_state=ego_state,
                                   perceived_road=perceived_road)

    enriched_state_serialized = enriched_state.serialize()
    print(enriched_state_serialized)
    state_module.DDS.publish(topic='StateModulePub::StateWriter', data=enriched_state_serialized)

    time.sleep(1)


