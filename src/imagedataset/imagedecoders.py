
from typing import Optional, Tuple
from PIL import Image
from torchvision import transforms as T
import numpy as np
import io
from .operations import ImageDecoder
import cv2
from .interpolations import (
           Interpolation, 
           interpolation_is_accimage, 
           interpolation_is_opencv, 
           interpolation_is_pil
)

# Optional imports
try:
    import accimage
except Exception:
    accimage = None

try:
    import turbojpeg
except Exception:
    turbojpeg = None


# DECODERS
class DecoderPIL(ImageDecoder):

    def __init__(self, 
                 size: Optional[Tuple[int, int]] = None, 
                 interpolation: Optional[Interpolation] = Interpolation.CV_AREA):
        """
            Initializes the PIL decocer. If size is not None a resize will be performed.

            Args:
                size (tuple, optional): the target size (for resizing).
                interpolation (Interpolation, optional): the interpolation mode
                (for resizing).
        """
        super().__init__(size, interpolation)

        self.size      = size
        self.interpolation = interpolation.value

        self.is_opencv = False if self.size is None \
                               else interpolation_is_opencv(interpolation)
        self.is_pil    = False if self.size is None \
                               else interpolation_is_pil(interpolation)
        self.is_accimage = False if self.size is None \
                               else interpolation_is_accimage(interpolation)

        if self.is_accimage:
            raise ValueError(f"{self.__class__.__name__} does not support " + 
                              "ACCIMAGE interpolation!")


    def __call__(self, bytes: bytes) -> np.ndarray:

        """
            Decodes an image from bytes and returns a `np.ndarray`.
        """

        img = Image.open(io.BytesIO(bytes))
        img = img.convert('RGB')
        if self.is_opencv:                    
            img = np.asarray(img)
            img = cv2.resize(img, self.size, interpolation=self.interpolation)
        elif self.is_pil:
            img = img.resize(self.size, resample=self.interpolation)
            img = np.asarray(img)
        else:
            img = np.asarray(img)
            
        return img


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

        self.size      = size
        self.interpolation = interpolation.value

        self.is_opencv = False if self.size is None \
                               else interpolation_is_opencv(interpolation)
        self.is_pil    = False if self.size is None \
                               else interpolation_is_pil(interpolation)
        self.is_accimage = False if self.size is None \
                               else interpolation_is_accimage(interpolation)

        if self.is_pil:
            raise ValueError(f"{self.__class__.__name__} does not support " + 
                              "PIL interpolation!")
        if self.is_accimage:
            raise ValueError(f"{self.__class__.__name__} does not support " + 
                              "ACCIMAGE interpolation!")


    def __call__(self, bytes: bytes) -> np.ndarray:

        """
            Decodes an image from bytes and returns a `np.ndarray`.
        """
        img = np.fromstring(bytes, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)

        if self.is_opencv:
            img = cv2.resize(img, self.size, self.interpolation)  

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img


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

        self.size      = size
        self.interpolation = interpolation.value

        self.is_opencv = False if self.size is None \
                               else interpolation_is_opencv(interpolation)
        self.is_pil    = False if self.size is None \
                               else interpolation_is_pil(interpolation)
        self.is_accimage = False if self.size is None \
                               else interpolation_is_accimage(interpolation)

        if self.is_pil:
            raise ValueError(f"{self.__class__.__name__} does not support " + 
                              "PIL interpolation!")

    def __call__(self, bytes: bytes) -> np.ndarray:

        """
            Decodes an image from bytes and returns a `np.ndarray`.
        """

        img = accimage.Image(bytes)

        if self.is_accimage:
            img.resize(size=self.size)

        img_np = np.empty([img.channels, img.height, img.width], dtype=np.uint8)
        img.copyto(img_np)
        img = np.transpose(img_np, (1, 2, 0))

        if self.is_opencv:
            img = cv2.resize(img, self.size, self.interpolation) 

        return img


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

        self.size      = size
        self.interpolation = interpolation.value

        self.is_opencv = False if self.size is None \
                               else interpolation_is_opencv(interpolation)
        self.is_pil    = False if self.size is None \
                               else interpolation_is_pil(interpolation)
        self.is_accimage = False if self.size is None \
                               else interpolation_is_accimage(interpolation)

        if self.is_pil:
            raise ValueError(f"{self.__class__.__name__} does not support " + 
                              "PIL interpolation!")
        if self.is_accimage:
            raise ValueError(f"{self.__class__.__name__} does not support " + 
                              "ACCIMAGE interpolation!")

        self.jpeg = turbojpeg.TurboJPEG()


    def __call__(self, bytes: bytes) -> np.ndarray:

        """
            Decodes an image from bytes and returns a `np.ndarray`.
        """

        img = self.jpeg.decode(bytes, pixel_format=turbojpeg.TJPF_RGB)
        if self.is_opencv:
            img = cv2.resize(img, self.size, interpolation=self.interpolation) 
        return img
