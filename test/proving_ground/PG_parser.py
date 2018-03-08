import numpy as np
import matplotlib.pyplot as plt
import math
import os

plt.switch_backend('Qt5Agg')

"""
This script is used to analyze DM log files (such as AV_Log_dm_main.log) and plot
the agent velocity and acceleration along with the error in latitude and longitude
calculated in the "if" section in the Trajectory Planner. This check was originally
meant to check the control response to the change in the required velocity and acceleration.
"""


def plot_dynamics_and_if_errors(log_path):

    v_agent = []
    a_agent = []
    lon_if_error = []
    lat_if_error = []

    f = open(log_path, 'r')

    while True:
        text = f.readline()
        if not text:
            break
        if '\'trajectory\': {\'array\': [' in text:
            v_agent.append(float(text.split('\'trajectory\': {\'array\': [')[1].split(', ')[9]))
            a_agent.append(float(text.split('\'trajectory\': {\'array\': [')[1].split(', ')[10]))
        elif ('lon_lat_errors: [' in text) and ('TrajectoryPlanningFacade' in text or 'TrajectoryPlanningSimulationFacade' in text):
            lon_if_error.append(float(text.split('lon_lat_errors: [ ')[1].split()[0]))
            lat_if_error.append(float(text.split('lon_lat_errors: [ ')[1].split()[1]))
        elif 'NoValidTrajectoriesFound' in text:
            v_agent.append(0)
            a_agent.append(0)

    v_agent = np.asarray(v_agent)
    a_agent = np.asarray(a_agent)
    lon_if_error = np.asarray(lon_if_error)
    lat_if_error = np.asarray(lat_if_error)

    fig = plt.figure()
    p1 = fig.add_subplot(411)
    plt.title('velocity')
    p2 = fig.add_subplot(412)
    plt.title('acc')
    p3 = fig.add_subplot(413)
    plt.title('lon_if_error')
    p4 = fig.add_subplot(414)
    plt.title('lat_if_error')

    p1.plot(v_agent)
    p2.plot(a_agent)
    p3.plot(lon_if_error)
    p4.plot(lat_if_error)
    plt.show()


if __name__ == "__main__":
    # Enter path of log file to analyze here:
    plot_dynamics_and_if_errors('/home/yzystl/av_code/spav/logs/AV_Log_dm_main.log')
