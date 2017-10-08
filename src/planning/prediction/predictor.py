import numpy as np
from decision_making.src.state.state import DynamicObject


class Predictor:
    """
    Base class for predictors of dynamic objects
    """

    @staticmethod
    def predict(dynamic_object: DynamicObject,  predicted_timestamps: np.ndarray) -> np.ndarray:
        """

        :param dynamic_object:
        :param predicted_timestamps:
        :return: predicted object's locations in global map coordinates np.array([x, y, theta, vel])
        """

