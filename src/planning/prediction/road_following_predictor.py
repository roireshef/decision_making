from decision_making.src.messages.navigation_plan_message import NavigationPlanMsg
from decision_making.src.planning.prediction.constants import LOOKAHEAD_MARGIN_DUE_TO_ROUTE_LINEARIZATION_APPROXIMATION
from decision_making.src.planning.prediction.predictor import Predictor
from decision_making.src.state.state import DynamicObject
from mapping.src.model.map_api import MapAPI
import numpy as np

from mapping.src.transformations.geometry_utils import CartesianFrame


class RoadFollowingPredictor(Predictor):
    """
    See base class
    """

def predict(dynamic_object: DynamicObject, predicted_timestamps: np.ndarray, map_api: MapAPI,
            nav_plan:NavigationPlanMsg) -> np.ndarray:
    # we assume the object is travelling exactly on a constant latitude. (i.e., lateral speed = 0)
    object_velocity = np.linalg.norm([dynamic_object.v_x, dynamic_object.v_y])

    # we assume the objects is travelling with a constant velocity, therefore the lookahead distance is
    lookahead_distance = (predicted_timestamps[-1] - dynamic_object.timestamp) * object_velocity
    lookahead_distance += LOOKAHEAD_MARGIN_DUE_TO_ROUTE_LINEARIZATION_APPROXIMATION

    lookahead_route = map_api.get_lookahead_points(dynamic_object.road_localization.road_id,
                                                   dynamic_object.road_localization.road_lon,
                                                   lookahead_distance,
                                                   dynamic_object.road_localization.full_lat,
                                                   nav_plan)

    # resample the route to predicted_timestamps
    predicted_distances_from_start = object_velocity * (predicted_timestamps - dynamic_object.timestamp)
    CartesianFrame.resample_curve(curve=lookahead_route, arbitrary_curve_sampling_points=predicted_distances_from_start)
