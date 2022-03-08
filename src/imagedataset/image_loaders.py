
from typing import Callable, Any, Optional, Tuple
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms as T
import numpy as np
import io


def get_loader_by_name(name: str, size: Optional[Tuple] = None) -> Callable[[str], Any]:
    """ Returns the loader with the given name. """

    name = name.lower()

    if name == "pil":
        return get_pil_loader(size)
    elif name == "accimage":
        return get_accimage_loader(size)
    elif name == "opencv":
        return get_cv2_loader(size)
    elif name == "turbojpeg":
        return get_turbojpeg_loader(size)
    else:
        ValueError(f"Loader {name} not implemented!")


def get_decoder_by_name(name: str) -> Callable[[bytes], Any]:
    """ Returns the decoder with the given name. """

    name = name.lower()

    if name == "pil":
        return get_pil_decoder()
    elif name == "accimage":
        return get_accimage_decoder()
    elif name == "opencv":
        return get_cv2_decoder()
    elif name == "turbojpeg":
        return get_turbojpeg_decoder()
    else:
        ValueError(f"Decoder {name} not implemented!") 


# Loaders: functions to load images from disk

# PIL
def get_pil_loader(size: Optional[Tuple] = None, 
                   pil_resize: Optional[bool] = False,
                   cv_resize: Optional[bool] = True,
                   torch_resize: Optional[bool] = False,
                   interpolation: Optional[Any] = None) -> Callable[[str], np.ndarray]:

    assert sum([pil_resize, cv_resize, torch_resize]) == 1, \
           "One and only one resize should be True."

    if cv_resize:
        import cv2
        if interpolation is None:
            interpolation = cv2.INTER_LINEAR_EXACT
    elif torch_resize:
        if interpolation is None:
            interpolation = InterpolationMode.BILINEAR
 
        if size is not None:
            resize = T.Resize(list(size), interpolation)
    else:
        if interpolation is None:
            interpolation = Image.BILINEAR

    def pil_loader(path: str) -> np.ndarray:

        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            if size is not None:
                if not cv_resize and not torch_resize: # PIL resize
                    img = img.resize(size, interpolation)
                    img = np.asarray(img)
                elif cv_resize:                        # CV resize
                    img = np.asarray(img)
                    img = cv2.resize(img, size, interpolation)
                else:                                  # TORCH resize
                    img = resize(img)
            return img

    return pil_loader


# ACCIMAGE: check https://github.com/pytorch/accimage/blob/master/test.py
def get_accimage_loader(size: Optional[Tuple] = None) -> Callable[[str], np.ndarray]:
    import accimage

    def accimage_loader(path: str) -> Image.Image:
        image = accimage.Image(path)
        if size is not None:
            image.resize(size=size)
        image_np = np.empty([image.channels, image.height, image.width], dtype=np.uint8)
        image.copyto(image_np)
        image_np = np.transpose(image_np, (1, 2, 0))
        return image_np

    return accimage_loader


# OPEN CV
def get_cv2_loader(size: Optional[Tuple] = None, 
                   cv_interpolation: Optional[int] = None) \
                   -> Callable[[str], np.ndarray]:

    import cv2

    if cv_interpolation is None:
        cv_interpolation = cv2.INTER_LINEAR_EXACT

    def cv2_loader(path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if size is not None:
            img = cv2.resize(img, size, cv_interpolation)        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    return cv2_loader

# TURBO JPEG
def get_turbojpeg_loader(size: Optional[Tuple] = None, \
                         cv_interpolation: Optional[int] = None) \
                         -> Callable[[str], np.ndarray]:

    from turbojpeg import TurboJPEG, TJPF_RGB
    import cv2

    if cv_interpolation is None:
        cv_interpolation = cv2.INTER_LINEAR_EXACT

    jpeg = TurboJPEG()

    def turbojpg_loader(path: str) -> np.ndarray: 
        with open(path, mode="rb") as f:
            data = f.read()
        img = jpeg.decode(data, pixel_format=TJPF_RGB)
        if size is not None:
            img = cv2.resize(img, size, cv_interpolation) 
        return img

    return turbojpg_loader


# decoders:  functions to decode images from bytes.

# PIL
def get_pil_decoder() -> Callable[[bytes], Image.Image]:
    def pil_decoder(bytes_image: bytes) -> Image.Image:
        img = Image.open(io.BytesIO(bytes_image))
        img = img.convert('RGB')
        return img

    return pil_decoder


# OPEN CV
def get_cv2_decoder() -> Callable[[bytes], np.ndarray]:
    import cv2

    def cv2_decoder(bytes_image: bytes) -> np.ndarray: 
        img = np.fromstring(bytes_image, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    
        return img

    return cv2_decoder

# JPEG TURBO
def get_turbojpeg_decoder() -> Callable[[bytes], np.ndarray]:
    from turbojpeg import TurboJPEG, TJPF_RGB

    jpeg = TurboJPEG()

    def turbojpg_decoder(bytes_image: bytes) -> np.ndarray: 
        return jpeg.decode(bytes_image, pixel_format=TJPF_RGB)
 
    return turbojpg_decoder

# ACCIMAGE: check https://github.com/pytorch/accimage/blob/master/test.py
def get_accimage_decoder() -> Callable[[bytes], np.ndarray]:
    import accimage

    def accimage_decoder(bytes_image: bytes) -> np.ndarray:
        return accimage.Image(bytes_image)

    return accimage_decoder


