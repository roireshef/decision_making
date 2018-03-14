from decision_making.test.lcm_trajectory_player.lcmpubsub import LcmPubSub


def create_pubsub(config_file="lcm_socket_config.json", pubSubType=LcmPubSub, domain_participant=None, domain_id=0):
    if pubSubType == LcmPubSub:
        return LcmPubSub(config_file, domain_id)

    raise TypeError("pubSubType=%s is unsupported" % pubSubType)
