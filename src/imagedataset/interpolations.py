import cv2
from PIL import Image
from enum import Enum


class Interpolation(Enum):

    """
        NOTE (cv2 documentation):
        To shrink an image, it will generally look best with CV_INTER_AREA interpolation, 
        whereas to enlarge an image, it will generally look best with CV_INTER_CUBIC 
        (slow) or CV_INTER_LINEAR (faster but still looks OK).
    """
    CV_BILINEAR = cv2.INTER_LINEAR
    CV_BILINEAR_EXACT = cv2.INTER_LINEAR_EXACT
    CV_NEAREST = cv2.INTER_NEAREST
    CV_BICUBIC = cv2.INTER_CUBIC
    CV_LANCZOS = cv2.INTER_LANCZOS4
    CV_AREA = cv2.INTER_AREA

    PIL_NEAREST = Image.NEAREST
    PIL_BICUBIC = Image.BICUBIC
    PIL_BILINEAR = Image.BILINEAR
    PIL_LANCZOS = Image.LANCZOS

    ACCIMAGE_BUILDIN = "ACCIMAGE"


def interpolation_is_opencv(interpolation: Enum) -> bool:
    """ Returns True for OpenCV interpolations. """
    return interpolation.name[0:2] == 'CV'

def interpolation_is_pil(interpolation: Enum) -> bool:
    """ Returns True for PIL interpolations. """
    return interpolation.name[0:3] == 'PIL'

def interpolation_is_accimage(interpolation: Enum) -> bool:
    """ Returns True for AccImage interpolations. """
    return interpolation.name[0:3] == 'ACC'