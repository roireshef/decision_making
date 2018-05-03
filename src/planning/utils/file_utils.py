from numpy import frombuffer


class BinaryReadWrite:
    """
    Saves and loads arrays and matrices to a binary file
    """
    @staticmethod
    def save(array, file_path):
        """
        Creates and opens a binary file, and writes a multi-dimensional array to it
        :param array:
        :param file_path:
        :return:
        """
        with open(file_path, 'wb+') as fh:
            fh.write(bytearray(array))

    @staticmethod
    def load(file_path, shape):
        """
        Opens a file, loads a multi-dimensional array from it, and reshapes it to the desired shape.
        :param file_path:
        :param shape:
        :return:
        """
        with open(file_path, 'rb') as fh:
            return frombuffer(fh.read(), dtype='float64').reshape(shape)
