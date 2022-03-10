from typing import Any, Tuple
import numpy as np


class Operation:
    """ Abstract class for any operation. """

    def __init__(self):
        pass

    def __call__(self, input: Any) -> Any:
        """ Take an input, compute the operation and returns the output. """
        raise NotImplementedError(f"{self.__class__.__name__} is an abstract class.")


class ByteLoader(Operation):
    """ Class to load bytes. """

    def __init__(self):
        super().__init__()

    def __call__(self, path: str) -> bytes:
        """ Take a path, read file and return bytes. """
        with open(path, 'rb') as f:
            data = f.read() 
        return data


class FileLoader(Operation):
    """ Abstract class for FileLoaders. """

    def __init__(self):
        super().__init__()

    def __call__(self, path: str) -> np.ndarray:
        """ Take a path, read file and convert to np.ndarray. """
        super().__call__(path)


class FileDecoder(Operation):
    """ Abstract class for FileDecoder. """

    def __init__(self):
        super().__init__()

    def __call__(self, bytes: bytes) -> np.ndarray:
        """ Take bytes as input and convert to np.ndarray. """
        super().__call__(bytes)


class ImageLoader(FileLoader):
    """ Abstract class for a FileLoader of images. """

    def __init__(self, size: Tuple[int, int], interpolation: Any):
        """
            Args:
                size (Tuple): the size of image (after resize).
                interpolation (Any): the interpolation mode (for resize).
        """

        super().__init__()
        self.size = size
        self.interpolation = interpolation

    def __call__(self, path: str) -> np.ndarray:
        """ Take a path, read file and convert to np.ndarray. """
        super().__call__(path)


class ImageDecoder(FileDecoder):
    """ Abstract class for a FileDecoder of images. """

    def __init__(self, size: Tuple[int, int], interpolation: Any):
        """
            Args:
                size (Tuple): the size of image (after resize).
                interpolation (Any): the interpolation mode (for resize).
        """

        super().__init__()
        self.size = size
        self.interpolation = interpolation

    def __call__(self, bytes: bytes) -> np.ndarray:
        """ Take bytes as input and convert to np.ndarray. """
        super().__call__(bytes)
