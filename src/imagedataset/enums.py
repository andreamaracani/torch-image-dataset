import cv2
from PIL import Image
import torch
import numpy as np
from enum import Enum


class Interpolation(Enum):

    """
        NOTE (cv2 documentation):
        To shrink an image, it will generally look best with CV_INTER_AREA interpolation, 
        whereas to enlarge an image, it will generally look best with CV_INTER_CUBIC 
        (slow) or CV_INTER_LINEAR (faster but still looks OK).
    """
                        #  VALUE                SOURCE           NAME
    CV_BILINEAR       = (cv2.INTER_LINEAR,       "CV",          "BILINEAR")
    CV_BILINEAR_EXACT = (cv2.INTER_LINEAR_EXACT, "CV",          "BILINEAR_EXACT")
    CV_NEAREST        = (cv2.INTER_NEAREST,      "CV",          "NEAREST")
    CV_BICUBIC        = (cv2.INTER_CUBIC,        "CV",          "BICUBIC")
    CV_LANCZOS        = (cv2.INTER_LANCZOS4,     "CV",          "LANCZOS")
    CV_AREA           = (cv2.INTER_AREA,         "CV",          "INTRA_AREA")
    PIL_NEAREST       = (Image.NEAREST,          "PIL",         "NEAREST")
    PIL_BICUBIC       = (Image.BICUBIC,          "PIL",         "BICUBIC")
    PIL_BILINEAR      = (Image.BILINEAR,         "PIL",         "BILINEAR")
    PIL_LANCZOS       = (Image.LANCZOS,          "PIL",         "LANCZOS")
    ACCIMAGE_BUILDIN  = (0,                      "ACCIMAGE",    "ACCIMAGE")


def interpolation_is_opencv(interpolation: Interpolation) -> bool:
    """ Returns True for OpenCV interpolations. """
    return interpolation.value[1] == "CV"

def interpolation_is_pil(interpolation: Interpolation) -> bool:
    """ Returns True for PIL interpolations. """
    return interpolation.value[1] == "PIL"

def interpolation_is_accimage(interpolation: Interpolation) -> bool:
    """ Returns True for AccImage interpolations. """
    return interpolation.value[1] == "ACCIMAGE"


class OutputFormat(Enum):
    """
        Outputs formats:
            SHAPE - DTYPE - CONVERSION
        
        SHAPES: 
            - NCHW (channels first)
            - NHWC (channels last)
            - UNALTERED (as loaded)
        
        DTYPES:
            - FLOAT16
            - FLOAT32
            - UINT8
            - INT32
            - INT64

        CONVERSIONS:
            - TENSOR
            - NUMPY
    """
                            # channels_first, dtype, to_tensor 

    # float16
    NCHW_FLOAT16_TENSOR      = (True,  torch.float16, True)
    NCHW_FLOAT16_NUMPY       = (True,  np.float16,   False)
    NHWC_FLOAT16_TENSOR      = (False, torch.float16, True)
    NHWC_FLOAT16_NUMPY       = (False, np.float16,   False)
    UNALTERED_FLOAT16_TENSOR = (False, torch.float16, True)
    UNALTERED_FLOAT16_NUMPY  = (False, np.float16,   False)

    # float32
    NCHW_FLOAT32_TENSOR      = (True,  torch.float32, True)
    NCHW_FLOAT32_NUMPY       = (True,  np.float32,   False)
    NHWC_FLOAT32_TENSOR      = (False, torch.float32, True)
    NHWC_FLOAT32_NUMPY       = (False, np.float32,   False)
    UNALTERED_FLOAT32_TENSOR = (False, torch.float32, True)
    UNALTERED_FLOAT32_NUMPY  = (False, np.float32,   False)

    # uint8
    NCHW_UNIT8_TENSOR        = (True,  torch.uint8,   True)
    NCHW_UNIT8_NUMPY         = (True,  np.uint8,     False)
    NHWC_UNIT8_TENSOR        = (False, torch.uint8,   True)
    NHWC_UNIT8_NUMPY         = (False, np.uint8,     False)
    UNALTERED_UNIT8_TENSOR   = (False, torch.uint8,   True)
    UNALTERED_UNIT8_NUMPY    = (False, np.uint8,     False)

    # int32
    NCHW_INT32_TENSOR        = (True,  torch.int32,   True)
    NCHW_INT32_NUMPY         = (True,  np.int32,     False)
    NHWC_INT32_TENSOR        = (False, torch.int32,   True)
    NHWC_INT32_NUMPY         = (False, np.int32,     False)
    UNALTERED_INT32_TENSOR   = (False, torch.int32,   True)
    UNALTERED_INT32_NUMPY    = (False, np.int32,     False)

    # int64
    NCHW_INT64_TENSOR        = (True,  torch.int64,   True)
    NCHW_INT64_NUMPY         = (True,  np.int64,     False)
    NHWC_INT64_TENSOR        = (False, torch.int64,   True)
    NHWC_INT64_NUMPY         = (False, np.int64,     False)
    UNALTERED_INT64_TENSOR   = (False, torch.int64,   True)
    UNALTERED_INT64_NUMPY    = (False, np.int64,     False)
