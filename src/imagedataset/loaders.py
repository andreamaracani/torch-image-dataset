from typing import Callable

from torchvision.transforms.functional import convert_image_dtype
from torchvision.transforms import ToTensor
from PIL import Image
from torchvision.io.image import read_image, decode_image
import torch
import io
import numpy as np
import cv2


def get_torch_loader() -> Callable[[str], torch.Tensor]:
    """
        Returns the default loader.
    """
    def default_loader(path: str) -> torch.Tensor:
        return convert_image_dtype(read_image(path), torch.float32)

    return default_loader


def get_pil_loader() -> Callable[[str], torch.Tensor]:
    """
        Returns the pil loader.
    """
    def pil_loader(path: str) -> torch.Tensor:

        toTensor = ToTensor()

        with open(path, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
        image_tensor = toTensor(image)
        return image_tensor

    return pil_loader


def get_turbojpg_loader() -> Callable[[str], torch.Tensor]:
    import numpy as np
    from turbojpeg import TurboJPEG, TJPF_RGB

    jpeg = TurboJPEG()

    def turbojpg_loader(path: str) -> torch.Tensor: 
        with open(path, mode="rb") as f:
            data = f.read()

        image_array = jpeg.decode(data, pixel_format=TJPF_RGB)
        image_array = np.rollaxis(image_array, 2, 0)  
        return torch.from_numpy(image_array)

    return turbojpg_loader


# decoders
def get_torch_decoder() -> Callable[[bytes], torch.Tensor]:

    def torch_decoder(bytes_image: bytes) -> torch.Tensor:
        image_array = np.fromstring(bytes_image, np.uint8)
        image_tensor = torch.from_numpy(image_array)
        image_tensor = decode_image(image_tensor)
        return convert_image_dtype(image_tensor, torch.float32)

    return torch_decoder


def get_pil_decoder() -> Callable[[bytes], torch.Tensor]:

    toTensor = ToTensor()
    def pil_decoder(bytes_image: bytes) -> torch.Tensor:
        image = Image.open(io.BytesIO(bytes_image))
        image = image.convert('RGB')
        image_tensor = toTensor(image)
        return image_tensor

    return pil_decoder


def get_cv2_decoder() -> Callable[[bytes], torch.Tensor]:

    def cv2_decoder(bytes_image: bytes) -> torch.Tensor: 
        image_array = np.fromstring(bytes_image, np.uint8)
        image_array = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)     
        image_array = np.moveaxis(image_array, 2, 0) 
        image_tensor = torch.from_numpy(image_array)
        return convert_image_dtype(image_tensor, torch.float32)

    return cv2_decoder


def get_turbojpg_decoder() -> Callable[[bytes], torch.Tensor]:
    from turbojpeg import TurboJPEG, TJPF_RGB

    jpeg = TurboJPEG()

    def turbojpg_decoder(bytes_image: bytes) -> torch.Tensor: 
        image_array = jpeg.decode(bytes_image, pixel_format=TJPF_RGB)
        image_array = np.rollaxis(image_array, 2, 0)  
        image_tensor = torch.from_numpy(image_array)
        return convert_image_dtype(image_tensor, torch.float32)
        
    return turbojpg_decoder

