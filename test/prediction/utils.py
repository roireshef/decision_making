import numpy as np

from decision_making.src.state.state import NewDynamicObject


class Utils:

    @staticmethod
    def assert_dyn_objects_numerical_fields_are_equal(actual_object: NewDynamicObject, expected_object: NewDynamicObject) -> None:
        """
        Assert that all fields with numerical values are equal between actual and expected object
        :param actual_object:
        :param expected_object:
        :return:
        """

        assert np.all(np.isclose(actual_object.cartesian_state, expected_object.cartesian_state, atol=1e-2))
        assert np.all(np.isclose(actual_object.map_state.road_fstate, expected_object.map_state.road_fstate, atol=1e-2))

        assert np.isclose(actual_object.map_state.road_id, expected_object.map_state.road_id, atol=1e-2)
        assert np.isclose(actual_object.obj_id, expected_object.obj_id, atol=1e-2)
        assert np.isclose(actual_object.timestamp, expected_object.timestamp, atol=1e-2)
        assert np.isclose(actual_object.confidence, expected_object.confidence, atol=1e-2)
