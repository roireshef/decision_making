from types import FunctionType
from typing import List, Any
from copy import copy

class FunctionUtils:
    """This class contains helper functions that provide information about given functions"""

    @staticmethod
    def get_argument_annotations(func: FunctionType) -> List[Any]:
        """
        Returns the type hints given to a function's arguments. In other words, if an
        annotation is given to the function's return value, it will not be included.

        :param func: function in question
        :return: a list containing the type hint for each function argument
        """
        annotations = copy(func.__annotations__)

        if 'return' in annotations:
            annotations.pop('return')

        return list(annotations.values())
