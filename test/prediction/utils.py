import numpy as np

from decision_making.src.state.state import DynamicObject


class Utils:

    @staticmethod
    def assert_dyn_objects_numerical_fields_are_equal(actual_object: DynamicObject, expected_object: DynamicObject) -> None:
        """
        Assert that all fields with numerical values are equal between actual and expected object
        :param actual_object:
        :param expected_object:
        :return:
        """

        assert np.all(np.isclose(actual_object.cartesian_state, expected_object.cartesian_state, atol=1e-2))
        assert np.all(np.isclose(actual_object.map_state.a_LaneFState, expected_object.map_state.a_LaneFState, atol=1e-2))

        assert np.isclose(actual_object.map_state.e_i_LaneID, expected_object.map_state.e_i_LaneID, atol=1e-2)
        assert np.isclose(actual_object.e_i_ObjectID, expected_object.e_i_ObjectID, atol=1e-2)
        assert np.isclose(actual_object.e_Cnt_Timestamp, expected_object.e_Cnt_Timestamp, atol=1e-2)
        assert np.isclose(actual_object.e_r_Confidence, expected_object.e_r_Confidence, atol=1e-2)
