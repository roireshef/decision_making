# This file stores column names for numpy matrices #

# Column indices for ego-state [x, y, theta, v]
EGO_X, EGO_Y, EGO_THETA, EGO_V = 0, 1, 2, 3

# Column indices for a route in cartesian-frame [x, y, yaw, curvature, curvature-tag]
R_X, R_Y, R_THETA, R_K, R_K_TAG = 0, 1, 2, 3, 4

# Column indices for frenet-frame [s, s-dot, s-dotdot, d, d-dot, d-dotdot]
F_SX, F_SV, F_SA, F_DX, F_DV, F_DA = 0, 1, 2, 3, 4, 5
