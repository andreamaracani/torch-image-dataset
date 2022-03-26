
from typing import Optional, Tuple
from PIL import Image
import numpy as np
import io
from .operations import ImageDecoder, ImageResizer
from .imageresizers import ResizerPIL, ResizerOpenCV
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
    jpeg = turbojpeg.TurboJPEG()
except Exception:
    turbojpeg = None
    jpeg = None


# DECODERS
class DecoderPIL(ImageDecoder):

    def __init__(self, 
                 size: Optional[Tuple[int, int]] = None, 
                 interpolation: Optional[Interpolation] = Interpolation.PIL_BILINEAR):
        """
            Initializes the PIL decocer. If size is not None a resize will be performed.

            Args:
                size (tuple, optional): the target size (for resizing).
                interpolation (Interpolation, optional): the interpolation mode
                (for resizing).
        """
        super().__init__(size, interpolation)

    def __repr__(self) -> str:
        interpolation_name = self.interpolation.name
        name = str(self._size) + ", " + interpolation_name \
               if self._size is not None else "(*, *)"
        return f"{self.__class__.__name__}{name}"


    def __call__(self, bytes: bytes) -> np.ndarray:

        """
            Decodes an image from bytes and returns a `np.ndarray`.
        """

        img = Image.open(io.BytesIO(bytes))
        img = img.convert('RGB')
        if interpolation_is_opencv(self._interpolation) and self._size is not None:                    
            img = np.asarray(img)
            img = cv2.resize(img, self.size, interpolation=self.interpolation.value[0])
        elif interpolation_is_pil(self._interpolation) and self._size is not None:
            img = img.resize(self.size, resample=self.interpolation.value[0])
            img = np.asarray(img)
        elif interpolation_is_accimage(self._interpolation) and self._size is not None:
            raise ValueError(f"{self.__class__.__name__} does not support " + 
                              "ACCIMAGE interpolation!")
        else:
            img = np.asarray(img)
            
        return img

    def resizer(self) -> ImageResizer:
        """ Get the resizer of this image loader. """

        if interpolation_is_accimage(self._interpolation):
            raise ValueError("Accimage resizer not available!")
        
        if interpolation_is_opencv(self._interpolation):
            return ResizerOpenCV(self._size, self._interpolation)

        if interpolation_is_pil(self._interpolation):
            return ResizerPIL(self._size, self._interpolation)

class DecoderOpenCV(ImageDecoder):

    def __init__(self, 
                 size: Optional[Tuple[int, int]] = None, 
                 interpolation: Optional[Interpolation] = Interpolation.CV_AREA):
        """
            Initializes the OpenCV decoder. If size is not None a resize will be 
            performed.

            Args:
                size (tuple, optional): the target size (for resizing).
                interpolation (Interpolation, optional): the interpolation mode
                (for resizing).
        """
        super().__init__(size, interpolation)


    def __repr__(self) -> str:
        interpolation_name = self.interpolation.name
        name = str(self._size) + ", " + interpolation_name \
               if self._size is not None else "(*, *)"
        return f"{self.__class__.__name__}{name}"


    def __call__(self, bytes: bytes) -> np.ndarray:

        """
            Decodes an image from bytes and returns a `np.ndarray`.
        """

        if interpolation_is_accimage(self._interpolation) and self._size is not None: 
            raise ValueError(f"{self.__class__.__name__} does not support " + 
                              "ACCIMAGE interpolation!")

        img = np.fromstring(bytes, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)

        if interpolation_is_opencv(self._interpolation) and self._size is not None: 
            img = cv2.resize(img, self.size, self.interpolation.value[0])  

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if interpolation_is_pil(self._interpolation) and self._size is not None: 
            img = Image.fromarray(img.astype('uint8'), 'RGB')
            img = img.resize(self.size, resample=self.interpolation.value[0])
            img = np.asarray(img)

        return img


    def resizer(self) -> ImageResizer:
        """ Get the resizer of this image loader. """

        if interpolation_is_accimage(self._interpolation):
            raise ValueError("Accimage resizer not available!")
        
        if interpolation_is_opencv(self._interpolation):
            return ResizerOpenCV(self._size, self._interpolation)

        if interpolation_is_pil(self._interpolation):
            return ResizerPIL(self._size, self._interpolation)


class DecoderAccImage(ImageDecoder):

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
        return f"{self.__class__.__name__}{name}"


    def __call__(self, bytes: bytes) -> np.ndarray:

        """
            Decodes an image from bytes and returns a `np.ndarray`.
        """

        img = accimage.Image(bytes)

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

    def resizer(self) -> ImageResizer:
        """ Get the resizer of this image loader. """

        if interpolation_is_accimage(self._interpolation):
            raise ValueError("Accimage resizer not available!")
        
        if interpolation_is_opencv(self._interpolation):
            return ResizerOpenCV(self._size, self._interpolation)

        if interpolation_is_pil(self._interpolation):
            return ResizerPIL(self._size, self._interpolation)


class DecoderTurboJPEG(ImageDecoder):

    def __init__(self, 
                 size: Optional[Tuple[int, int]] = None, 
                 interpolation: Optional[Interpolation] = Interpolation.CV_AREA):
        """
            Initializes the TurboJPEG decoder. If size is not None a resize will be 
            performed.

            Args:
                size (tuple, optional): the target size (for resizing).
                interpolation (Interpolation, optional): the interpolation mode
                (for resizing).
        """

        if turbojpeg is None:
            raise ValueError("TurboJpeg module not available!")
    
        super().__init__(size, interpolation)

    def __repr__(self) -> str:
        interpolation_name = self.interpolation.name
        name = str(self._size) + ", " + interpolation_name \
               if self._size is not None else "(*, *)"
        return f"{self.__class__.__name__}{name}"


    def __call__(self, bytes: bytes) -> np.ndarray:

        """
            Decodes an image from bytes and returns a `np.ndarray`.
        """
        if interpolation_is_accimage(self._interpolation) and self._size is not None: 
            raise ValueError(f"{self.__class__.__name__} does not support " + 
                              "ACCIMAGE interpolation!")

        img = jpeg.decode(bytes, pixel_format=turbojpeg.TJPF_RGB)

        if interpolation_is_opencv(self._interpolation) and self._size is not None: 
            img = cv2.resize(img, self.size, interpolation=self.interpolation.value[0]) 
        
        if interpolation_is_pil(self._interpolation) and self._size is not None: 
            img = Image.fromarray(img.astype('uint8'), 'RGB')
            img = img.resize(self.size, resample=self.interpolation.value[0])
            img = np.asarray(img)

        return img


    def resizer(self) -> ImageResizer:
        """ Get the resizer of this image loader. """

        if interpolation_is_accimage(self._interpolation):
            raise ValueError("Accimage resizer not available!")
        
        if interpolation_is_opencv(self._interpolation):
            return ResizerOpenCV(self._size, self._interpolation)

        if interpolation_is_pil(self._interpolation):
            return ResizerPIL(self._size, self._interpolation)