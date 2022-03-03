import heapq
from typing import Dict, Callable

import numpy as np
from decision_making.src.rl_agent.utils.numpy_utils import NumpyUtils


class HeapUtils:
    @staticmethod
    def ksmallest(k, heap):
        """ efficiently pull k-smallest elements from a heap. items are not guaranteed to be in order! """
        if k == 1:
            # o(1)
            item = next(iter(heap), None)
            return [item] if item else []
        elif k >= len(heap):
            # o(1)
            return heap
        else:
            # o(n*log(k))
            return heapq.nsmallest(k, heap)


class ListUtils:
    @staticmethod
    def slice_list(lst, mask):
        """ slice list according to mask """
        return [item for item, condition in zip(lst, mask) if condition]


class DictUtils:
    @staticmethod
    def recursive_merge_with(primary: Dict, secondary: Dict):
        """ override primary dictionary with fields in secondary dictionary recursively """
        return {k: primary[k] if k not in secondary else secondary[k] if not isinstance(primary.get(k), Dict) else
        DictUtils.recursive_merge_with(primary[k], secondary[k]) for k in set().union(primary, secondary)}


class VectorizedDict(dict):
    def __init__(self, *args, **kwargs):
        """
        A python-dict variant that sorts items by keys, and overrides the get-item functionality such that values can
        be queried in a vectorized fashion (with a vector of indices) efficiently by leveraging the fact keys were
        already sorted in the constructor
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)

        # create numpy object-arrays of keys and values
        self.k = NumpyUtils.from_list_of_tuples(list(self.keys()))
        self.v = NumpyUtils.from_list_of_tuples(list(self.values()))

        # sort by key
        sidx = self.k.argsort()
        self.k = self.k[sidx]
        self.v = self.v[sidx]

    def __getitem__(self, keys):
        if isinstance(keys, tuple):
            return self.get(keys)

        item_array = np.empty(len(keys), dtype=object)
        item_array[:] = keys
        return self.v[np.searchsorted(self.k, item_array)]


class UpdateableDict(dict):
    def __init__(self, *args, **kwargs):
        """
        A python-dict variant that exposes a method that takes a lambda function the user can specify that maps
        the original dictionary to an additional fields to add to it based on its values
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)

    def update_with(self, f: Callable[[Dict], Dict]):
        """
        f(self) is a lambda functoin that maps between the original dictionary and the new portion to add to it
        For example:
            d = UpdateableDict({"x": 3, "y": 5})
            >> {'x': 3, 'y': 5}

            d.update_with(lambda org: {"z": org["x"] * org["y"]})
            >> {'x': 3, 'y': 5, 'z': 15}
        :param f:
        :return:
        """
        return UpdateableDict(self, **f(self))



