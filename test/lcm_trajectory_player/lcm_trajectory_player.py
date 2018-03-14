from decision_making.test.lcm_trajectory_player.lcmpubsub import LcmPubSub
from decision_making.test.lcm_trajectory_player.pubsub_factory import create_pubsub
from decision_making.test.lcm_trajectory_player.trajectory_plan_message import TrajectoryPlanMsg
import time
import numpy as np
TRAJECTORY_TOPIC = "TRAJECTORY_LCM"
LCM_SOCKET_CONFIG = 'lcm_socket_config.json'
def main():
    pubsub = create_pubsub(LCM_SOCKET_CONFIG, LcmPubSub)
    trajectory = read_trajectory('trajectories/proving_ground_center_lane.txt')
    traj_msg = TrajectoryPlanMsg(timestamp=0, trajectory=trajectory, current_speed=5.0)
    for i in range(0, 1000):
        # sending the trajectory message
        pubsub.publish(TRAJECTORY_TOPIC, traj_msg.serialize())
        time.sleep(0.2)


def read_trajectory(filename):
    trajectory = []
    file = open(filename, 'r')
    for line in file.readlines():
        trajectory.append([float(val) for val in line.split(',')])
    return np.array(trajectory)

if __name__ == '__main__':
    main()
