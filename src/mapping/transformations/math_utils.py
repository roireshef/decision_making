from typing import Union, TypeVar

import numpy as np

DIVISION_FLOATING_ACCURACY = 10 ** -10

class Math:
    T = TypeVar('T', bound=Union[float, np.ndarray])

    @staticmethod
    def div(a,  b, precision=DIVISION_FLOATING_ACCURACY):
        # type: (T,  T, float) -> T
        """
        divides a/b with desired floating-point precision
        """
        div = np.divide(a, b).astype(np.int_)
        mod = np.subtract(a, np.multiply(div, b))
        add_ones = 1 * (np.fabs(mod - b) < precision)

        if isinstance(div, np.ndarray):
            return (div + add_ones).astype(np.int_)
        else:
            return int(div + add_ones)

    @staticmethod
    def mod(a,  b, precision=DIVISION_FLOATING_ACCURACY):
        # type: (T,  T, float ) -> T
        """
        modulo a % b with desired floating-point precision
        """
        div = np.divide(a, b).astype(np.int_)
        mod = np.subtract(a, np.multiply(div, b))

        return b * (np.fabs(mod - b) < precision) + mod * (np.fabs(mod - b) > precision) * (np.fabs(mod) > precision)
