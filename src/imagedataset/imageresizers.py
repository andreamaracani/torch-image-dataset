
from typing import Optional, Tuple
from PIL import Image
from torchvision import transforms as T
import numpy as np
from .operations import ImageResizer
import cv2
from .enums import (Interpolation, 
                    interpolation_is_accimage, 
                    interpolation_is_opencv, 
                    interpolation_is_pil)

# Optional imports
try:
    import accimage
except Exception:
    accimage = None

try:
    import turbojpeg
except Exception:
    turbojpeg = None


# RESIZERS
class ResizerPIL(ImageResizer):

    def __init__(self, 
                 size: Optional[Tuple[int, int]] = None, 
                 interpolation: Optional[Interpolation] = Interpolation.PIL_BILINEAR):
        """
            Initializes the PIL resizer.

            Args:
                size (tuple, optional): the target size (for resizing).
                interpolation (Interpolation, optional): the interpolation mode
                (for resizing).
        """
        super().__init__(size, interpolation)
    
    def __repr__(self) -> str:
        interpolation_name = self.interpolation.value[2]
        size = str(self._size) if self._size is not None else "(*, *)"
        return f"{self.__class__.__name__}[{interpolation_name}, {size}]"

    def __call__(self, array: np.ndarray) -> np.ndarray:

        """ Resizes the numpy array. """

        if interpolation_is_pil(self._interpolation) and self._size is not None:
            img = Image.fromarray(array.astype('uint8'), 'RGB')
            img = img.resize(self.size, resample=self.interpolation.value[0])
            array = np.asarray(img)
        elif self._size is None:
            array = array
        else:
            raise ValueError("Interpolation not supported!")

        return array


class ResizerOpenCV(ImageResizer):

    def __init__(self, 
                 size: Optional[Tuple[int, int]] = None, 
                 interpolation: Optional[Interpolation] = Interpolation.CV_BICUBIC):
        """
            Initializes the PIL resizer.

            Args:
                size (tuple, optional): the target size (for resizing).
                interpolation (Interpolation, optional): the interpolation mode
                (for resizing).
        """
        super().__init__(size, interpolation)

    def __repr__(self) -> str:
        interpolation_name = self.interpolation.value[2]
        size = str(self._size) if self._size is not None else "(*, *)"
        return f"{self.__class__.__name__}[{interpolation_name}, {size}]"


    def __call__(self, array: np.ndarray) -> np.ndarray:
        """ Resizes the numpy array. """
        
        if interpolation_is_opencv(self._interpolation) and self._size is not None:
            array = cv2.resize(array, self.size, self.interpolation.value[0])  
        elif self._size is None:
            array = array
        else:
            raise ValueError("Interpolation not supported!")

        return array
