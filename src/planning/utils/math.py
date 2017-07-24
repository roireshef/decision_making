from src.planning.global_constants import DIVISION_FLOATING_ACCURACY
import math


def div(a: float,  b: float, precision: float = DIVISION_FLOATING_ACCURACY):
    """
    divides a/b with desired floating-point precision
    """
    div, mod = divmod(a, b)
    if math.fabs(mod - b) < precision:
        return int(div + 1)
    else:
        return int(div)


def mod(a,  b, precision: float = DIVISION_FLOATING_ACCURACY):
    """
    modulo a % b with desired floating-point precision
    """
    div, mod = divmod(a, b)
    if math.fabs(mod - b) < precision:
        return 0
    else:
        return mod
