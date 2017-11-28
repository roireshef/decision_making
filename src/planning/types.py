import numpy as np

## CARTESIAN FRAME ##
CartesianPoint = np.ndarray  # [C_X, C_Y]
CartesianPath = np.ndarray  # a numpy matrix having rows of CartesianPoint

CartesianState = np.ndarray  # [C_X, C_Y, C_THETA, C_V]
CartesianTrajectory = np.ndarray  # a numpy matrix having rows of CartesianState
CartesianTrajectories = np.ndarray  # a tensor of CartesianTrajectory

CartesianExtendedState = np.ndarray  # [C_X, C_Y, C_THETA, C_V, C_A, C_K]
CartesianExtendedTrajectory = np.ndarray  # A Cartesian-Frame trajectory: a numpy matrix of CartesianExtendedState
CartesianExtendedTrajectories = np.ndarray  # Cartesian-Frame trajectories: a tensor of CartesianExtendedTrajectory

C_X, C_Y, C_THETA, C_V, C_A, C_K = 0, 1, 2, 3, 4, 5  # Column indices for cartesian-state [x, y, theta, v, a, k]


Curve = np.ndarray  # [CURVE_X, CURVE_Y, CURVE_THETA]
ExtendedCurve = np.ndarray  # [CURVE_X, CURVE_Y, CURVE_THETA, CURVE_K, CURVE_K_TAG]

# [x, y, yaw, (1st derivative of yaw), curvature-tag (2nd derivative of yaw)]
CURVE_X, CURVE_Y, CURVE_THETA, CURVE_K, CURVE_K_TAG = 0, 1, 2, 3, 4

## FRENET FRAME ##
FrenetPoint = np.ndarray  # [FP_SX, FP_DX]

FP_SX, FP_DX = 0, 1  # frenet-frame position only [sx, dx]

FrenetState = np.ndarray  # [FS_SX, FS_SV, FS_SA, FS_DX, FS_DV, FS_DA]
FrenetTrajectory = np.ndarray  # A Frenet-Frame trajectory: a numpy matrix of FrenetState
FrenetTrajectories = np.ndarray  # Frenet-Frame trajectories: a tensor of FrenetTrajectory

FS_SX, FS_SV, FS_SA, FS_DX, FS_DV, FS_DA = 0, 1, 2, 3, 4, 5  # frenet-frame state: [s, s-dot, s-dotdot, d, d-dot, d-dotdot]


## MISC ##
# A (two-cells) 1D numpy array represents limits (min, max)
Limits = np.ndarray
LIMIT_MIN = 0
LIMIT_MAX = 1