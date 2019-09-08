import numpy as np
import sys

## TIMESTAMPS ##
GlobalTimeStamp = int                   # global timestamp in [nanosec] since given time
GlobalTimeStampInSec = float            # global timestamp in [sec] since given time
# Initialization of timestamps in the past: these constants assure that each real timstamp will be greater
MinGlobalTimeStamp = -sys.maxsize       # past timestamps in [nanosec]
MinGlobalTimeStampInSec = -np.inf       # past timestamp in [sec]

## CARTESIAN FRAME ##

# [C_X, C_Y]
CartesianPoint2D = np.ndarray
# a numpy matrix having rows of CartesianPoint2D [:, [C_X, C_Y]]
CartesianPath2D = np.ndarray
# a numpy tensor having rows of CartesianPath2D [:, :, [C_X, C_Y]]
CartesianPaths2D = np.ndarray
# a numpy matrix having rows of vector in 2D cartesian space [:, [d(C_X), d(C_Y)]]
CartesianVectors2D = np.ndarray
# Arbitrarily-shaped tesnor with last dimension being CartesianPoint2D [..., [C_X, C_Y]]
CartesianPointsTensor2D = np.ndarray
# Arbitrarily-shaped tesnor with last dimension being a vector in 2D cartesian space [..., [d(C_X), d(C_Y)]]
CartesianVectorsTensor2D = np.ndarray

# [C_X, C_Y, C_Z]
CartesianPoint3D = np.ndarray
# a numpy matrix having rows of CartesianPoint3D [:, [C_X, C_Y, C_Z]]
CartesianPath3D = np.ndarray

# [C_X, C_Y, C_YAW, C_V]
CartesianState = np.ndarray
# a numpy matrix having rows of CartesianState [:, [C_X, C_Y, C_YAW, C_V]]
CartesianTrajectory = np.ndarray
# a tensor of CartesianTrajectory [:, :, [C_X, C_Y, C_YAW, C_V]]
CartesianTrajectories = np.ndarray

# [C_X, C_Y, C_YAW, C_V, C_A, C_K]
CartesianExtendedState = np.ndarray
# A Cartesian-Frame trajectory: a numpy matrix of CartesianExtendedState [:, [C_X, C_Y, C_YAW, C_V, C_A, C_K]]
CartesianExtendedTrajectory = np.ndarray
# Cartesian-Frame trajectories: a tensor of CartesianExtendedTrajectory
CartesianExtendedTrajectories = np.ndarray

# Column indices for cartesian-state [x, y, yaw, velocity, acceleration, curvature]
C_X, C_Y, C_YAW, C_V, C_A, C_K = 0, 1, 2, 3, 4, 5
C_Z = 2

# [CURVE_X, CURVE_Y, CURVE_YAW]
Curve = np.ndarray
# [CURVE_X, CURVE_Y, CURVE_YAW, CURVE_K, CURVE_K_TAG]
ExtendedCurve = np.ndarray

# [x, y, yaw, (1st derivative of yaw), curvature-tag (2nd derivative of yaw)]
CURVE_X, CURVE_Y, CURVE_YAW, CURVE_K, CURVE_K_TAG = 0, 1, 2, 3, 4

## FRENET FRAME ##
FrenetPoint = np.ndarray  # [FP_SX, FP_DX]

# frenet-frame position only [sx, dx]
FP_SX, FP_DX = 0, 1

# [FS_SX, FS_SV, FS_SA, FS_DX, FS_DV, FS_DA]
FrenetState2D = np.ndarray
# A Frenet-Frame trajectory: a numpy matrix of FrenetState2D [:, [FS_SX, FS_SV, FS_SA, FS_DX, FS_DV, FS_DA]]
FrenetStates2D = FrenetTrajectory2D = np.ndarray
# Frenet-Frame trajectories: a tensor of FrenetTrajectory2D [:, :, [FS_SX, FS_SV, FS_SA, FS_DX, FS_DV, FS_DA]]
FrenetTrajectories2D = np.ndarray

# Frenet-frame state: [s, s-dot, s-dotdot, d, d-dot, d-dotdot]
FS_SX, FS_SV, FS_SA, FS_DX, FS_DV, FS_DA = 0, 1, 2, 3, 4, 5

# [FS_X, FS_V, FS_A]
FrenetState1D = np.ndarray
# A Frenet-Frame trajectory: a numpy matrix of FrenetState1D [:, [FS_X, FS_V, FS_A]]
FrenetStates1D = FrenetTrajectory1D = np.ndarray
# Frenet-Frame trajectories: a tensor of FrenetTrajectory1D [:, :, [FS_X, FS_V, FS_A]]
FrenetTrajectories1D = np.ndarray

# frenet-frame 1D state: [x, x-dot, x-dotdot]
FS_X, FS_V, FS_A, = 0, 1, 2
FS_1D_LEN = 3
FS_2D_LEN = 6


CRT_LEN = 6


# [s,d] Polynomials
S5, S4, S3, S2, S1, S0, D5, D4, D3, D2, D1, D0 = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11

## MISC ##
# A (two-cells) 1D numpy array represents limits (min, max)
Limits = np.ndarray
LIMIT_MIN = 0
LIMIT_MAX = 1

# BehavioralGridState cell tuple-indices
LAT_CELL, LON_CELL = 0, 1

# 1D Numpy array of indices (dtype = np.int)
NumpyIndicesArray = np.ndarray

# boolean array
BoolArray = np.ndarray

# array of ActionSpec
ActionSpecArray = np.array

# Info on road signs
SIGN_TYPE, SIGN_S = 0, 1
