from typing import List

from decision_making.src.planning.types import CartesianTrajectory, C_V, C_YAW, C_X, C_Y
from decision_making.src.prediction.predictor import Predictor
from decision_making.src.state.state import DynamicObject
import numpy as np


class TestPredictorMock(Predictor):
    def predict_object(self, dynamic_object: DynamicObject,
                       prediction_timestamps: np.ndarray) -> np.ndarray:
        traj: CartesianTrajectory = np.array([[0.0, 0.0, np.pi / 4, x] for x in range(len(prediction_timestamps))])
        traj[:, C_X] = np.cumsum(traj[:, C_V] * np.cos(traj[:, C_YAW]))
        traj[:, C_Y] = np.cumsum(traj[:, C_V] * np.sin(traj[:, C_YAW]))

        return traj

    def predict_object_on_road(self, dynamic_object: DynamicObject, prediction_timestamps: np.ndarray) -> List[
        DynamicObject]:
        pass
