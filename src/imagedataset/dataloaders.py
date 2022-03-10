"""
    Inspired from APEX example at:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py#L256

    and also from Timm loader at:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/loader.py

"""
from types import SimpleNamespace
from typing import Callable, List, Optional, Any

import torch
from torch.utils.data import DataLoader

import numpy as np
import random



def get_collate(memory_format: Optional[Any] = torch.channels_last, 
                images_last2first: Optional[bool] = True, 
                to_tensor: Optional[bool] = False) -> Callable[[Any], SimpleNamespace]:
    """
        Get the collate function.

        Args:
            memory_format (torch.channel_last|torch.contiguous_format): the data memory
            format for output torch.Tensors.
            images_last2first (bool, optional): True to change the shape of images from
            (H x W x C) to (C x H x W).
            to_tensor (bool, optional): True to convert np.uint8 into Tensor

        Returns:
            Callable, the collate function.
    """
    return lambda batch: collate_function(batch, 
                                          images_last2first, 
                                          to_tensor, 
                                          memory_format)

def collate_function(batch: List[SimpleNamespace],
                     images_last2first: bool,
                     to_tensor: bool, 
                     memory_format) -> SimpleNamespace:
    """
        The collate function.

        Args:
            batch (list[SimpleNamespace]): the input batch.
            images_last2first (bool, optional): True to change the shape of images from
            (H x W x C) to (C x H x W).
            to_tensor (bool, optional): True to convert np.uint8 into Tensor.
            memory_format (torch.channel_last|torch.contiguous_format): the data memory
            format for output torch.Tensors.

        Returns:
            a SimpleNamespace with collated fields.
        
        NOTE:
            current types supported inside batch's SimpleNamespaces:
            - int, np.integer
            - None
            - np.ndarray
    """

    # get keys of SimpleNamespace
    keys = batch[0].__dict__.keys()
    batch_size = len(batch)
    collate_dict = {}

    for k in keys:
        
        first_value = batch[0].__dict__[k]
        
        # CASE NONE
        if first_value is None:
            out_value = [None for _ in batch]

        # CASE INT
        elif isinstance(first_value, int) or isinstance(first_value, np.integer):
            if to_tensor:
                out_value = torch.tensor([sample.__dict__[k] for sample in batch], 
                                          dtype=torch.int64)
            else:
                out_value = np.array([sample.__dict__[k] for sample in batch], 
                                      dtype=np.uint8)
        # CASE NDARRAY
        elif isinstance(first_value, np.ndarray):
            values = [sample.__dict__[k] for sample in batch]

            shape = values[0].shape

            if len(shape) == 3 and images_last2first:
                new_shape = (batch_size, shape[2], shape[0], shape[1])
            else:
                new_shape = tuple([batch_size] + list(shape))

            if to_tensor:
                out_value = torch.zeros(new_shape, dtype=torch.uint8) \
                             .contiguous(memory_format=memory_format)
            else:
                out_value = np.zeros(shape=new_shape, dtype=np.uint8)

            for i, value in enumerate(values):
                
                if len(shape) == 3 and images_last2first:
                    value = np.rollaxis(value, 2)

                if to_tensor:
                    value = torch.from_numpy(value)

                out_value[i] += value

        # OTHER TYPES
        else:
            msg = f"Collate of targets {type(first_value)} not implemented!"
            raise NotImplementedError(msg)

        collate_dict[k] = out_value

    return SimpleNamespace(**collate_dict)


class CudaLoader:
    """
        A dataloader with prefatching that loads all data to gpu. 
        It can converts float32 to float16 and it can do the normalization operation.    
    """
    def __init__(self,
                 loader: DataLoader,
                 mean: Optional[tuple] = None,
                 std: Optional[tuple] = None,
                 fp16=False):
        """
            Args:
                loader (torch.utils.data.Dataloader): the dataloader.
                mean (tuple, optional): the mean to subtract (only to images).
                std (tuple, optional): the std to divide (only to images).
                fp16 (bool, optional): True to convert tensors to half precision.
        """

        self.dataloader = loader

        self.mean = None
        self.std  = None
        self.normalize = False

        if mean is not None and std is not None:
            self.mean = torch.tensor([x for x in mean]).cuda().view(1, 3, 1, 1)
            self.std = torch.tensor([x for x in std]).cuda().view(1, 3, 1, 1)
            self.normalize = True

        self.fp16 = fp16

        if fp16 and self.normalize:
            self.mean = self.mean.half()
            self.std = self.std.half()

    def __iter__(self):

        stream = torch.cuda.Stream()
        first = True

        for batch in self.dataloader:
            
            next_input_dict = {}
            with torch.cuda.stream(stream):
                
                for k in batch.__dict__.keys():
                    next_input = batch.__dict__[k]
                    next_input = next_input.cuda(non_blocking=True)

                    if k == "image":
                        if self.fp16 and self.normalize:
                            next_input = next_input.half().sub_(self.mean).div_(self.std)
                        elif self.fp16:
                            next_input = next_input.half()
                        elif self.normalize:
                            next_input = next_input.float().sub_(self.mean).div_(self.std)
                        else:
                            next_input = next_input.float()

                    next_input_dict[k] = next_input

            if not first:
                yield SimpleNamespace(**input_dict)
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input_dict = next_input_dict.copy()

        yield SimpleNamespace(**input_dict)

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset


    def _worker_init(worker_id, worker_seeding='all'):
        worker_info = torch.utils.data.get_worker_info()
        assert worker_info.id == worker_id

        if isinstance(worker_seeding, Callable):
            seed = worker_seeding(worker_info)
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed % (2 ** 32 - 1))

        else:
            assert worker_seeding in ('all', 'part')
            # random / torch seed already called in dataloader iter class w/ 
            # worker_info.seed to reproduce some old results (same seed + hparam combo), 
            # partial seeding is required (skip numpy re-seed)

            if worker_seeding == 'all':
                np.random.seed(worker_info.seed % (2 ** 32 - 1))



