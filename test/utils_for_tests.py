from decision_making.src.planning.types import CartesianExtendedTrajectory
import numpy as np


class Utils:

    @staticmethod
    def read_trajectory(filename: str) -> CartesianExtendedTrajectory:
        trajectory = []
        file = open(filename, 'r')
        for line in file.readlines():
            trajectory.append([float(val) for val in line.split(',')])
        return np.array(trajectory)