from __future__ import annotations

from typing import Any, Tuple, Optional
import numpy as np
from .enums import Interpolation


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

    def __init__(self, size: Tuple[int, int], interpolation: Interpolation):
        """
            Args:
                size (Tuple): the size of image (after resize).
                interpolation (Interpolation): the interpolation mode (for resizing).
        """

        super().__init__()
        self._size = size
        self._interpolation = interpolation

    def __call__(self, path: str) -> np.ndarray:
        """ Take a path, read file and convert to np.ndarray. """
        super().__call__(path)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[{self._size}]"

    @property
    def interpolation(self) -> Interpolation:
        """ Get the interpolation of the current ImageLoader. """
        return self._interpolation
    
    @property
    def size(self) -> Tuple[int, int]:
        """ Get the size of the current ImageLoader. """
        return self._size
    
    @size.setter
    def size(self, size: Tuple[int, int]):
        """ Set the size for the current ImageLoader. """
        self._size = size

    def decoder(self, keep_resizer: Optional[bool] = False) -> ImageDecoder:
        """
            Get the corresponding decoder of this ImageLoader.

            Args:
                keep_resizer (bool, optional): True to keep the resizer also in the 
                decoder, False to remove the current resizer in the decoder.
        """
        raise NotImplementedError(f"{self.__class__.__name__} is an abstract class.")

    def resizer(self) -> ImageResizer:
        """ Get the corresponding image resizer of this ImageLoader. """
        raise NotImplementedError(f"{self.__class__.__name__} is an abstract class.")


class ImageDecoder(FileDecoder):
    """ Abstract class for a FileDecoder of images. """

    def __init__(self, size: Tuple[int, int], interpolation: Interpolation):
        """
            Args:
                size (Tuple): the size of image (after resize).
                interpolation (Any): the interpolation mode (for resizing).
        """

        super().__init__()
        self._size = size
        self._interpolation = interpolation

    @property
    def interpolation(self) -> Interpolation:
        """ Get the interpolation of the current ImageDecoder. """
        return self._interpolation
    
    @property
    def size(self) -> Tuple[int, int]:
        """ Get the size of the current ImageDecoder. """
        return self._size

    @size.setter
    def size(self, size: Tuple[int, int]):
        """ Set the size for the current ImageDecoder. """
        self._size = size

    def resizer(self) -> ImageResizer:
        """ Get the corresponding image resizer of this ImageDecoder. """
        raise NotImplementedError(f"{self.__class__.__name__} is an abstract class.")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[{self._size}]"

    def __call__(self, bytes: bytes) -> np.ndarray:
        """ Take bytes as input and convert to np.ndarray. """
        super().__call__(bytes)



class ImageResizer(Operation):
    """ Abstract class for an image resizer. """

    def __init__(self, size: Tuple[int, int], interpolation: Interpolation):
        """
            Args:
                size (Tuple): the size of image (after resize).
                interpolation (Any): the interpolation mode (for resizing).
        """

        super().__init__()
        self._size = size
        self._interpolation = interpolation

    @property
    def interpolation(self) -> Interpolation:
        """ Get the interpolation of the current ImageDecoder. """
        return self._interpolation
    
    @property
    def size(self) -> Tuple[int, int]:
        """ Get the size of the current ImageDecoder. """
        return self._size

    @size.setter
    def size(self, size: Tuple[int, int]):
        """ Set the size for the current ImageDecoder. """
        self._size = size

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[{self._size}]"

    def __call__(self, bytes: np.ndarray) -> np.ndarray:
        """ Take a np.ndarray as input and convert to np.ndarray. """
        super().__call__(bytes)