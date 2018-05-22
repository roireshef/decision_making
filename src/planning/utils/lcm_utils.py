import numpy as np

from common_data.lcm.generatedFiles.gm_lcm.LcmNumpyArray import LcmNumpyArray


class LCMUtils:
    @staticmethod
    def numpy_array_to_lcm_numpy_array(numpy_array):
        # type: (np.ndarray) -> LcmNumpyArray
        lcm_numpy_array = LcmNumpyArray()
        lcm_numpy_array.num_dimensions = len(numpy_array.shape)
        lcm_numpy_array.shape = list(numpy_array.shape)
        lcm_numpy_array.length = numpy_array.size
        lcm_numpy_array.data = numpy_array.flat.__array__().tolist()
        return lcm_numpy_array
