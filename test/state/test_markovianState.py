from common_data.lcm.config import config_defs, pubsub_topics
from common_data.lcm.python.Communication.lcmpubsub import LcmPubSub
from common_data.src.communication.pubsub.pubsub_factory import create_pubsub
from decision_making.src.state.state import State, EgoState
from decision_making.test.planning.custom_fixtures import state_with_history
import time

def test_markovianState_egoWithHistory_serDeserSuccessfully(state_with_history: State):
    pubsub = create_pubsub(config_defs.LCM_SOCKET_CONFIG, LcmPubSub)
    pubsub.subscribe(topic=pubsub_topics.STATE_TOPIC)

    state = state_with_history
    state_serialized = state.serialize()
    pubsub.publish(topic=pubsub_topics.STATE_TOPIC, data=state_serialized)

    time.sleep(0.2)

    received_state_str = pubsub.get_latest_sample(topic=pubsub_topics.STATE_TOPIC)
    state_deserialized = State.deserialize(received_state_str)

    assert len(state_deserialized.ego_state.history) == 9
