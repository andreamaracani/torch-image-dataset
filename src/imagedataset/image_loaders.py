
from typing import Callable, Any, Tuple
from torchvision.transforms.functional import convert_image_dtype
from PIL import Image
from torchvision.io.image import read_image
from torchvision import transforms as T
import torch
import io
import numpy as np


def get_loader_by_name(name: str) -> Callable[[str], Any]:
    """ Returns the loader with the given name. """

    name = name.lower()

    if name == "pil":
        return get_pil_loader()
    elif name == "accimage":
        return get_accimage_loader()
    elif name == "opencv":
        return get_cv2_loader()
    elif name == "torch":
        return get_torch_loader()
    elif name == "turbojpeg":
        return get_turbojpeg_loader()
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


def get_loader_resize_by_name(name: str, target_size: Tuple) -> Callable[[Any], Any]:
    """ Returns the loader with the given name. """

    name = name.lower()

    if name == "pil" or name=="torch":
        return T.Resize(target_size)
    elif name == "accimage":
        return lambda x: x.resize(size=target_size)
    elif name == "opencv":
        import cv2
        return lambda x: cv2.resize(x, target_size)
    elif name == "turbojpeg":
        msg = "Native resizing with turbojpeg not implemented right now."
        raise NotImplementedError(msg)
    else:
        ValueError(f"Loader {name} not implemented!")


# Loaders: functions to load images from disk

# PIL
def get_pil_loader() -> Callable[[str], Image.Image]:

    def pil_loader(path: str) -> Image.Image:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    return pil_loader

# ACCIMAGE: check https://github.com/pytorch/accimage/blob/master/test.py
def get_accimage_loader() -> Callable[[str], Image.Image]:
    import accimage

    def accimage_loader(path: str) -> Image.Image:
        return accimage.Image(path)

    return accimage_loader

# OPEN CV
def get_cv2_loader() -> Callable[[str], np.ndarray]:
    import cv2

    def cv2_loader(path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    return cv2_loader

# TORCH
def get_torch_loader() -> Callable[[str], torch.Tensor]:

    def torch_loader(path: str) -> torch.Tensor:
        return read_image(path)

    return torch_loader

# TURBO JPEG
def get_turbojpeg_loader() -> Callable[[str], np.ndarray]:
    from turbojpeg import TurboJPEG, TJPF_RGB

    jpeg = TurboJPEG()

    def turbojpg_loader(path: str) -> np.ndarray: 
        with open(path, mode="rb") as f:
            data = f.read()
        return jpeg.decode(data, pixel_format=TJPF_RGB)

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


