
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import cv2
from torchvision import transforms as T

from .operations import ImageDecoder, ImageLoader, ImageResizer
from .imagedecoders import DecoderPIL, DecoderOpenCV, DecoderTurboJPEG, DecoderAccImage
from .imageresizers import ResizerPIL, ResizerOpenCV
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
    jpeg = turbojpeg.TurboJPEG()
except Exception:
    jpeg = None

 
class LoaderPIL(ImageLoader):

    def __init__(self, 
                 size: Optional[Tuple[int, int]] = None, 
                 interpolation: Optional[Interpolation] = Interpolation.PIL_BILINEAR):
        """
            Initializes the PIL loader. If size is not None a resizing will be performed.

            Args:
                size (tuple, optional): the target size (for resizing).
                interpolation (Interpolation, optional): the interpolation mode
                (for resizing).
        """
        super().__init__(size, interpolation)


    def __repr__(self) -> str:
        interpolation_name = self._interpolation.name
        name = str(self._size) + ", " + interpolation_name \
               if self._size is not None else "(*, *)"

        return f"{self.__class__.__name__}[{name}]"


    def decoder(self, keep_resizer: bool) -> DecoderPIL:
        """ 
            Get the decoder of this image loader. 
            
            Args:
                keep_resizer (bool): True to keep the resize operation even in the 
                decoder.
        """
        size = self.size if keep_resizer else None
        return DecoderPIL(size, self.interpolation)


    def resizer(self) -> ImageResizer:
        """ Get the resizer of this image loader. """

        if interpolation_is_accimage(self._interpolation):
            raise ValueError("Accimage resizer not available!")
        
        if interpolation_is_opencv(self._interpolation):
            return ResizerOpenCV(self._size, self._interpolation)

        if interpolation_is_pil(self._interpolation):
            return ResizerPIL(self._size, self._interpolation)


    def __call__(self, path: str) -> np.ndarray:

        """ Loads the image at `path` and returns a `np.ndarray`. """

        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            if interpolation_is_opencv(self._interpolation) and self._size is not None:                    
                img = np.asarray(img)
                img = cv2.resize(img, self.size, interpolation=self.interpolation.value[0])
            elif interpolation_is_pil(self._interpolation) and self._size is not None: 
                img = img.resize(self.size, resample=self.interpolation.value[0])
                img = np.asarray(img)
            elif self._size is None:
                img = np.asarray(img)
            else:
                raise ValueError(f"{self.__class__.__name__} does not support " + 
                              "ACCIMAGE interpolation!")
            return img



class LoaderOpenCV(ImageLoader):

    def __init__(self, 
                 size: Optional[Tuple[int, int]] = None, 
                 interpolation: Optional[Interpolation] = Interpolation.CV_AREA):
        """
            Initializes the OpenCV loader. If size is not None a resizing will be 
            performed.

            Args:
                size (tuple, optional): the target size (for resizing).
                interpolation (Interpolation, optional): the interpolation mode
                (for resizing).
        """
        super().__init__(size, interpolation)

        if self.is_accimage:
            raise ValueError(f"{self.__class__.__name__} does not support " + 
                              "ACCIMAGE interpolation!")

    def __repr__(self) -> str:
        interpolation_name = self.interpolation.name
        name = str(self._size) + ", " + interpolation_name \
               if self._size is not None else "(*, *)"
        return f"{self.__class__.__name__}[{name}]"


    def decoder(self, keep_resizer: bool) -> DecoderOpenCV:
        """ 
            Get the decoder of this image loader. 
            
            Args:
                keep_resizer (bool): True to keep the resize operation even in the 
                decoder.
        """
        size = self.size if keep_resizer else None
        return DecoderOpenCV(size, self.interpolation)


    def resizer(self) -> ImageResizer:
        """ Get the resizer of this image loader. """

        if interpolation_is_accimage(self._interpolation):
            raise ValueError("Accimage resizer not available!")
        
        if interpolation_is_opencv(self._interpolation):
            return ResizerOpenCV(self._size, self._interpolation)

        if interpolation_is_pil(self._interpolation):
            return ResizerPIL(self._size, self._interpolation)


    def __call__(self, path: str) -> np.ndarray:

        """
            Loads the image at `path` (resizes it if necessary) and returns a 
            `np.ndarray`.
        """
        if interpolation_is_accimage(self._interpolation) and self._size is not None: 
            raise ValueError(f"{self.__class__.__name__} does not support " + 
                              "ACCIMAGE interpolation!")

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if interpolation_is_opencv(self._interpolation) and self._size is not None: 
            img = cv2.resize(img, self.size, self.interpolation.value[0])  

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if interpolation_is_pil(self._interpolation) and self._size is not None: 
            img = Image.fromarray(img.astype('uint8'), 'RGB')
            img = img.resize(self.size, resample=self.interpolation.value[0])
            img = np.asarray(img)

        return img


class LoaderAccImage(ImageLoader):

    def __init__(self, 
                 size: Optional[Tuple[int, int]] = None, 
                 interpolation: Optional[Interpolation] = Interpolation.CV_AREA):
        

        """ 
            Check https://github.com/pytorch/accimage/blob/master/test.py

            Initializes the AccImage loader. If size is not None a resize will be 
            performed.

            Args:
                size (tuple, optional): the target size (for resizing).
                interpolation (Interpolation, optional): the interpolation mode
                (for resizing).
        """

        if accimage is None:
            raise ValueError("AccImage module not available!")
    
        super().__init__(size, interpolation)


    def __repr__(self) -> str:
        interpolation_name = self.interpolation.name
        name = str(self._size) + ", " + interpolation_name \
               if self._size is not None else "(*, *)"
        return f"{self.__class__.__name__}[{name}]"


    def decoder(self, keep_resizer: bool) -> DecoderAccImage:
        """ 
            Get the decoder of this image loader. 
            
            Args:
                keep_resizer (bool): True to keep the resize operation even in the 
                decoder.
        """
        size = self.size if keep_resizer else None
        return DecoderAccImage(size, self.interpolation)


    def resizer(self) -> ImageResizer:
        """ Get the resizer of this image loader. """

        if interpolation_is_accimage(self._interpolation):
            raise ValueError("Accimage resizer not available!")
        
        if interpolation_is_opencv(self._interpolation):
            return ResizerOpenCV(self._size, self._interpolation)

        if interpolation_is_pil(self._interpolation):
            return ResizerPIL(self._size, self._interpolation)


    def __call__(self, path: str) -> np.ndarray:
        """
            Loads the image at `path` (resizes it if necessary) and returns a 
            `np.ndarray`.
        """

        img = accimage.Image(path)

        if interpolation_is_accimage(self._interpolation) and self._size is not None: 
            img.resize(size=self.size)

        img_np = np.empty([img.channels, img.height, img.width], dtype=np.uint8)
        img.copyto(img_np)
        img = np.transpose(img_np, (1, 2, 0))

        if interpolation_is_opencv(self._interpolation) and self._size is not None: 
            img = cv2.resize(img, self.size, self.interpolation.value[0]) 

        if interpolation_is_pil(self._interpolation) and self._size is not None: 
            img = Image.fromarray(img.astype('uint8'), 'RGB')
            img = img.resize(self.size, resample=self.interpolation.value[0])
            img = np.asarray(img)

        return img


class LoaderTurboJPEG(ImageLoader):

    def __init__(self, 
                 size: Optional[Tuple[int, int]] = None, 
                 interpolation: Optional[Interpolation] = Interpolation.CV_AREA):
        """
            Initializes the TurboJPEG loader. If size is not None a resize will be 
            performed.

            Args:
                size (tuple, optional): the target size (for resizing).
                interpolation (Interpolation, optional): the interpolation mode
                (for resizing).
        """

        if jpeg is None:
            raise ValueError("TurboJpeg module not available!")
    
        super().__init__(size, interpolation)
    
    def __repr__(self) -> str:
        interpolation_name = self.interpolation.name
        name = str(self._size) + ", " + interpolation_name \
               if self._size is not None else "(*, *)"
        return f"{self.__class__.__name__}[{name}]"


    def decoder(self, keep_resizer: bool) -> DecoderTurboJPEG:
        """ 
            Get the decoder of this image loader. 
            
            Args:
                keep_resizer (bool): True to keep the resize operation even in the 
                decoder.
        """
        size = self.size if keep_resizer else None
        return DecoderTurboJPEG(size, self._interpolation)


    def resizer(self) -> ImageResizer:
        """ Get the resizer of this image loader. """

        if interpolation_is_accimage(self._interpolation):
            raise ValueError("Accimage resizer not available!")
        
        if interpolation_is_opencv(self._interpolation):
            return ResizerOpenCV(self._size, self._interpolation)

        if interpolation_is_pil(self._interpolation):
            return ResizerPIL(self._size, self._interpolation)


    def __call__(self, path: str) -> np.ndarray:
        """
            Loads the image at `path` (resizes it if necessary) and returns a 
            `np.ndarray`.
        """
        if interpolation_is_accimage(self._interpolation) and self._size is not None: 
            raise ValueError(f"{self.__class__.__name__} does not support " + 
                              "ACCIMAGE interpolation!")

        with open(path, mode="rb") as f:
            data = f.read()
            img = jpeg.decode(data, pixel_format=turbojpeg.TJPF_RGB)

        if interpolation_is_opencv(self._interpolation) and self._size is not None: 
            img = cv2.resize(img, self.size, interpolation=self.interpolation.value[0])

        if interpolation_is_pil(self._interpolation) and self._size is not None: 
            img = Image.fromarray(img.astype('uint8'), 'RGB')
            img = img.resize(self.size, resample=self.interpolation.value[0])
            img = np.asarray(img)

        return img
