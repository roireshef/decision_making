from typing import List

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
            return frombuffer(fh.read(), dtype='uint8').reshape(shape)


class TextReadWrite:
    """
    Reads and writes list of strings as distinct text lines to .txt files
    """
    @staticmethod
    def write(lines_list: List[str], file_path):
        """
        Creates and opens a text file, and writes a list of lines to it
        :param lines_list: A list of strings to be written as distinct rows in txt file.
        :param file_path:
        :return:
        """
        with open(file_path, 'w') as fh:
            for line in lines_list:
                fh.write('%s\n' % line)

    @staticmethod
    def read(file_path):
        """
        Opens a text file, and reads lines of text from it into a list of strings.
        :param file_path:
        :return:
        """
        with open(file_path, 'r') as fh:
            lines = fh.readlines()
        return [line.split('\n')[0] for line in lines]
