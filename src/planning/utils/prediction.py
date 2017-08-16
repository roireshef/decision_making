import numpy as np

class Prediction:
    """
    predicting location & velocity for moving objects
    """

    @staticmethod
    def predict_dynamics(x: float, y: float, yaw: float, v_x: float, v_y: float, accel_lon: float, turn_radius: float,
                         dt: float) -> tuple((float, float, float, float, float)):
        """
        Predict the object's location, yaw and velocity after a given time period.
        The object may accelerate and move in circle with given radius.
        :param x: starting x in meters
        :param y: starting y in meters
        :param yaw: starting yaw in radians
        :param v_x: starting v_x in m/s
        :param v_y: starting v_y in m/s
        :param accel_lon: constant longitudinal acceleration in m/s^2
        :param turn_radius: in meters; positive CW, negative CCW, zero means straight motion
        :param dt: time period in seconds
        :return: goal x, y, yaw, v_x, v_y
        """
        sin_yaw = np.sin(yaw)
        cos_yaw = np.cos(yaw)
        start_vel = np.sqrt(v_x * v_x + v_y * v_y)
        # if the object will stop before goal_timestamp, then set dt to be until the stop time
        if accel_lon < -0.01 and accel_lon * dt < -start_vel:
            dt = start_vel / (-accel_lon)

        if turn_radius is not None and turn_radius != 0:  # movement by circle arc (not straight)
            # calc distance the object passes until goal_timestamp
            dist = start_vel * dt + 0.5 * accel_lon * dt * dt
            # calc yaw change (turn angle) in radians
            d_yaw = dist / turn_radius
            goal_yaw = yaw + d_yaw
            sin_next_yaw = np.sin(goal_yaw)
            cos_next_yaw = np.cos(goal_yaw)
            # calc the circle center
            circle_center = [x - turn_radius * sin_yaw, y + turn_radius * cos_yaw]
            # calc the end location
            goal_x = circle_center[0] + turn_radius * sin_next_yaw
            goal_y = circle_center[1] - turn_radius * cos_next_yaw
            # calc the end velocity
            end_vel = start_vel + accel_lon * dt
            goal_v_x = end_vel * cos_next_yaw
            goal_v_y = end_vel * sin_next_yaw
        else:  # straight movement
            acc_x = accel_lon * cos_yaw
            acc_y = accel_lon * sin_yaw
            goal_x = x + v_x * dt + 0.5 * acc_x * dt * dt
            goal_y = y + v_y * dt + 0.5 * acc_y * dt * dt
            goal_v_x = v_x + dt * acc_x
            goal_v_y = v_y + dt * acc_y
            goal_yaw = yaw

        return tuple((goal_x, goal_y, goal_yaw, goal_v_x, goal_v_y))
