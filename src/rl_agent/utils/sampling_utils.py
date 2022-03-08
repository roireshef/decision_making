from abc import abstractmethod, ABCMeta
from typing import Union, List

import numpy as np

from decision_making.src.global_constants import EPS
from decision_making.src.rl_agent.utils.class_utils import Representable


class Distribution(Representable, metaclass=ABCMeta):
    """ Abstract distribution class that user can sample from. Beneficial for implementing sampling environment
    parameters on a per-episode basis """
    @abstractmethod
    def sample(self) -> Union[float, int]:
        pass


class Constant(Distribution):
    """ Constant value distribution (deterministic) """
    def __init__(self, val: Union[float, int]):
        self.val = val

    def sample(self) -> Union[float, int]:
        return self.val


class CategoricalDistribution(Distribution):
    """ Categorical dsitribution that gets a list of values and a list of their probabilities """
    def __init__(self, values: List[Union[float, int]], probs: List[float] = None):
        probs = probs or [1/len(values)] * len(values)
        assert np.isclose(sum(probs), 1, rtol=1e-4), \
            'got probs=%s and it does sum up to %s instead of to 1' % (str(probs), sum(probs))

        self.values = values
        self.probs = probs

    def sample(self) -> Union[float, int]:
        return np.random.choice(self.values, p=self.probs)


class DiscreteUniformDistribution(CategoricalDistribution):
    """ Defines an integer range from <min_val> to <max_val> (inclusive) with resolution <res> """
    def __init__(self, min_val: Union[float, int], max_val: Union[float, int], res: Union[float, int]):
        super().__init__(list(np.arange(min_val, max_val+EPS, res)))


class ContinuousUniformDistribution(Distribution):
    """ Defines an continuous uniform distribution from <min_val> to <max_val> """
    def __init__(self, min_val: Union[float, int], max_val: Union[float, int]):
        self.min_val = min_val
        self.max_val = max_val

    def sample(self) -> float:
        return np.random.uniform(self.min_val, self.max_val)
