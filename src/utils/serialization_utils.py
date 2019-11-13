import numpy as np
from interface.Rte_Types.python.sub_structures.TsSYS_NonTypedSmallNumpyArray import TsSYSNonTypedSmallNumpyArray
from interface.Rte_Types.python.sub_structures.TsSYS_NonTypedNumpyArray import TsSYSNonTypedNumpyArray
from interface.Rte_Types.python.sub_structures.TsSYS_NonTypedIntNumpyArray import TsSYSNonTypedIntNumpyArray
from interface.Rte_Types.python.sub_structures.TsSYS_NumpyArray import TsSYSNumpyArray
from typing import Any


class SerializationUtils:
    @staticmethod
    def serialize_non_typed_small_array(arr: np.ndarray) -> TsSYSNonTypedSmallNumpyArray:
        ser_arr = TsSYSNonTypedSmallNumpyArray()
        ser_arr.e_Cnt_NumDimensions = len(arr.shape)
        ser_arr.a_Shape = list(arr.shape)
        ser_arr.e_Cnt_Length = arr.size
        ser_arr.a_Data = arr.ravel()
        
        return ser_arr

    @staticmethod
    def serialize_non_typed_array(arr: np.ndarray) -> TsSYSNonTypedNumpyArray:
        ser_arr = TsSYSNonTypedNumpyArray()
        ser_arr.e_Cnt_NumDimensions = len(arr.shape)
        ser_arr.a_Shape = list(arr.shape)
        ser_arr.e_Cnt_Length = arr.size
        ser_arr.a_Data = arr.ravel()

        return ser_arr

    @staticmethod
    def serialize_non_typed_int_array(arr: np.ndarray) -> TsSYSNonTypedIntNumpyArray:
        ser_arr = TsSYSNonTypedIntNumpyArray()
        ser_arr.e_Cnt_NumDimensions = len(arr.shape)
        ser_arr.a_Shape = list(arr.shape)
        ser_arr.e_Cnt_Length = arr.size
        ser_arr.a_Data = arr.ravel()

        return ser_arr

    @staticmethod
    def serialize_array(arr: np.ndarray) -> TsSYSNumpyArray:
        ser_arr = TsSYSNumpyArray()
        ser_arr.e_Cnt_NumDimensions = len(arr.shape)
        ser_arr.a_Shape = list(arr.shape)
        ser_arr.e_Cnt_Length = arr.size
        ser_arr.a_Data = arr.ravel()

        return ser_arr

    @staticmethod
    def deserialize_any_array(arr: Any) -> np.ndarray:
        """
        This method deseralizes any type of the above arrays.
        :param arr: Can be either TsSYSNonTypedSmallNumpyArray, TsSYSNonTypedNumpyArray, TsSYSNonTypedIntNumpyArray or TsSYSNumpyArray
        :return:
        """
        arr_shape = arr.a_Shape[:arr.e_Cnt_NumDimensions]
        arr_size = np.prod(arr_shape)

        return arr.a_Data[:arr_size].reshape(tuple(arr_shape))
