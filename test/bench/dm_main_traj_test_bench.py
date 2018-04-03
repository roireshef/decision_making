from common_data.lcm.config import config_defs
from common_data.lcm.config.pubsub_topics import PERCEIVED_SELF_LOCALIZATION_TOPIC
from common_data.lcm.generatedFiles.gm_lcm import LcmPerceivedSelfLocalization
from common_data.lcm.python.Communication.lcmpubsub import LcmPubSub
from common_data.src.communication.pubsub.pubsub_factory import create_pubsub
from decision_making.paths import Paths

test_fixed_trajectory = TP_MOCK_FIXED_TRAJECTORY_FILENAME = Paths.get_resource_absolute_path_filename(
        'fixed_trajectory_files/test_trajectory.txt')

def test_DMMainTraj_Bench_SingleLocalizationMessage_TrajectoryOutput():

    #Read first point in the test trajectory to accommodate trigger condition
    f = open(file=test_fixed_trajectory, mode='r')
    line = f.readline()
    line_arr = list(map(float, line.replace("\n", "").split(", ")))
    start_x, start_y = line_arr[0], line_arr[1]

    #Construct localization message
    localization_msg = LcmPerceivedSelfLocalization
    localization_msg.location.x = start_x
    localization_msg.location.y = start_y

    #create pubsub and send localization message
    pubsub = create_pubsub(config_defs.LCM_SOCKET_CONFIG, LcmPubSub)
    pubsub.publish(PERCEIVED_SELF_LOCALIZATION_TOPIC, localization_msg)

