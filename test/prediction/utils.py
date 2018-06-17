import numpy as np


class Utils:

    @staticmethod
    def assert_objects_numerical_fields_are_equal(actual_object, expected_object) -> None:
        """
        Assert that all fields with numerical values are equal between actual and expected object
        :param actual_object:
        :param expected_object:
        :return:
        """
        actual_fields = actual_object.__dict__
        expected_fields = expected_object.__dict__

        for field_name in actual_fields.keys():
            if issubclass(type(actual_fields[field_name]), np.ndarray):
                assert np.all(np.isclose(actual_fields[field_name], expected_fields[field_name], atol=1e-2))
            else:
                try:
                    assert np.isclose(actual_fields[field_name], expected_fields[field_name], atol=1e-2)
                except TypeError as e:
                    pass
