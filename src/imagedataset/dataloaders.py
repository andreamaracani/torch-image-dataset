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
import torch.distributed as dist

import numpy as np
import random

from .enums import OutputFormat


def get_collate(image_format: OutputFormat,
                label_format: OutputFormat,
                other_fields_format: OutputFormat, 
                memory_format: torch.memory_format) -> Callable[[Any], SimpleNamespace]:
    """
        Get the collate function.

        Args:
            image_format (OutputFormat): the output format for images. 
            label_format (OutputFormat): the output format for labels.
            other_fields_format (OutputFormat): the output format for other fields.
            memory_format (torch.memory_format): the data memory format for output 
            torch.Tensors.

        Returns:
            Callable, the collate function.
    """
    return lambda batch: collate_function(batch, 
                                          image_format=image_format,
                                          label_format=label_format,
                                          other_fields_format=other_fields_format,
                                          memory_format=memory_format)


def collate_function(batch: List[SimpleNamespace],
                     image_format: OutputFormat,
                     label_format: OutputFormat,
                     other_fields_format: OutputFormat,
                     memory_format: torch.memory_format) -> SimpleNamespace:

    """ 
        NOTE: the collate function can covert types and shape of inputs following the
        OutputFormats but it does not scales the values/range of tensors/arrays.
        Since images inputs are uint8 the tensors/arrays will be in range [0, 255] even
        if they are coverted to floats.

        Args:
            batch (list[SimpleNamespace]): the input batch.
            image_format (OutputFormat): the output format for images. 
            label_format (OutputFormat): the output format for labels.
            other_fields_format (OutputFormat): the output format for other fields.
            memory_format (torch.memory_format): the data memory format for output 
            torch.Tensors.

        Returns:
            a SimpleNamespace with collated fields.
        
    """

    # get keys of SimpleNamespace
    keys = batch[0].__dict__.keys()
    batch_size = len(batch)
    collate_dict = {}

    for k in keys:
        
        # take the correct output format
        if   k == "image":
            channels_first, dtype, to_tensor = image_format.value
        elif k == "label":
            channels_first, dtype, to_tensor = label_format.value
        else:
            channels_first, dtype, to_tensor = other_fields_format.value


        first_value = batch[0].__dict__[k]
        
        # CASE INT
        if isinstance(first_value, int) or isinstance(first_value, np.integer):
            if to_tensor:
                out_value = torch.tensor([sample.__dict__[k] for sample in batch], 
                                          dtype=dtype)
            else:
                out_value = np.array([sample.__dict__[k] for sample in batch], 
                                      dtype=dtype)
        # CASE NDARRAY
        elif isinstance(first_value, np.ndarray):
            values = [sample.__dict__[k] for sample in batch]

            shape = values[0].shape

            if len(shape) == 3 and channels_first:
                new_shape = (batch_size, shape[2], shape[0], shape[1])
            else:
                new_shape = tuple([batch_size] + list(shape))

            if to_tensor:
                out_value = torch.zeros(new_shape, dtype=dtype) \
                                 .contiguous(memory_format=memory_format)
            else:
                out_value = np.zeros(shape=new_shape, dtype=dtype)

            for i, value in enumerate(values):
                
                if len(shape) == 3 and channels_first:
                    value = np.rollaxis(value, 2)

                if to_tensor:
                    value = torch.from_numpy(value).to(dtype)

                out_value[i] += value

        # OTHER TYPES
        else:
            msg = f"Collate of targets {type(first_value)} not implemented!"
            raise NotImplementedError(msg)

        collate_dict[k] = out_value

    return SimpleNamespace(**collate_dict)


class CpuLoader:
    def __init__(self,
                 loader: DataLoader,
                 image_format: OutputFormat,
                 image_mean: Optional[tuple] = None,
                 image_std:  Optional[tuple] = None,
                 scale_image_floats: Optional[bool] = True):
        """
            Args:
                loader (torch.utils.data.Dataloader): the dataloader.
                mean (tuple, optional): the mean to subtract (only to images).
                std (tuple, optional): the std to divide (only to images).
                rank (int, optional): the local rank (device).
        """
        
        if "NCHW" not in image_format.name and "NHWC" not in image_format.name:
            raise ValueError("Images should be in NCHW or NHWC format.")

        if "TENSOR" not in image_format.name:
            raise ValueError("Images should be Tensors for the CpuLoader.")

        self.dtype = image_format.value[1]

        assert self.dtype == torch.float16 or self.dtype == torch.float32, \
        "OutputFormat for images should be float16 or float32!" 

        self.view  = [1, 3, 1, 1] if "NCHW" in image_format.name else [1, 1, 1, 3]

        # do we need to scale images?
        self.scale = scale_image_floats

        self.image_mean = image_mean
        self.image_std = image_std

        # do we need to normalize images?
        self.normalize = self.image_mean is not None and self.image_std is not None

        if self.scale:
            if self.normalize:
                self.image_mean = [x * 255. for x in self.image_mean]
                self.image_std  = [x * 255. for x in self.image_std]
            else:
                self.image_mean = [0., 0., 0.]
                self.image_std  = [255., 255., 255.]


        # dataloader
        self.dataloader    = loader
        self.sampler       = loader.sampler
        self.batch_sampler = loader.batch_sampler
        self.dataset       = loader.dataset

      
        if self.scale or self.normalize:
            self.image_mean = torch.tensor(self.image_mean).to(self.dtype)\
                              .view(self.view)
            self.image_std  = torch.tensor(self.image_std).to(self.dtype)\
                              .view(self.view)



    def __iter__(self):

        for batch in self.dataloader:

            if self.normalize or self.scale:
                batch.image = batch.image.to(self.dtype).sub_(self.image_mean)\
                                         .div_(self.image_std)
            else:
                batch.image = batch.image.to(self.dtype)

            yield batch


    def __len__(self):
        return len(self.loader)



class CudaLoader:
    """
        A dataloader with prefatching that loads all data to gpu. 
        It can converts float32 to float16 and it can do the normalization operation.    
    """
    def __init__(self,
                 loader: DataLoader,
                 image_format: OutputFormat,
                 image_mean: Optional[tuple] = None,
                 image_std:  Optional[tuple] = None,
                 scale_image_floats: Optional[bool] = True,
                 rank: Optional[int] = None):
        """
            Args:
                loader (torch.utils.data.Dataloader): the dataloader.
                mean (tuple, optional): the mean to subtract (only to images).
                std (tuple, optional): the std to divide (only to images).
                rank (int, optional): the local rank (device).
        """
        
        if "NCHW" not in image_format.name and "NHWC" not in image_format.name:
            raise ValueError("Images should be in NCHW or NHWC format.")

        if "TENSOR" not in image_format.name:
            raise ValueError("Images should be Tensors for the CudaLoader.")

        self.dtype = image_format.value[1]

        assert self.dtype == torch.float16 or self.dtype == torch.float32, \
        "OutputFormat for images should be float16 or float32!" 

        self.view  = [1, 3, 1, 1] if "NCHW" in image_format.name else [1, 1, 1, 3]

        # do we need to scale images?
        self.scale = scale_image_floats

        self.image_mean = image_mean
        self.image_std = image_std

        # do we need to normalize images?
        self.normalize = self.image_mean is not None and self.image_std is not None

        if self.scale:
            if self.normalize:
                self.image_mean = [x * 255. for x in self.image_mean]
                self.image_std  = [x * 255. for x in self.image_std]
            else:
                self.image_mean = [0., 0., 0.]
                self.image_std  = [255., 255., 255.]


        # dataloader
        self.dataloader    = loader
        self.sampler       = loader.sampler
        self.batch_sampler = loader.batch_sampler
        self.dataset       = loader.dataset

        # local rank 
        self.rank = rank

        # if None get local rank
        if self.rank is None:
            try:
                self.rank = dist.get_rank()
            except Exception:
                self.rank = 0
        
        # send std and mean to local rank
        if self.scale or self.normalize:
            self.image_mean = torch.tensor(self.image_mean).to(self.dtype)\
                              .to(self.rank).view(self.view)
            self.image_std  = torch.tensor(self.image_std).to(self.dtype)\
                              .to(self.rank).view(self.view)


    def __iter__(self):

        stream = torch.cuda.Stream(device=self.rank)
        first = True

        for batch in self.dataloader:
            
            next_input_dict = {}
            with torch.cuda.stream(stream):
                
                for k in batch.__dict__.keys():
                    next_input = batch.__dict__[k]
                    next_input = next_input.to(self.rank, non_blocking=True)

                    if k == "image":
                        if self.normalize or self.scale:
                            next_input = next_input.to(self.dtype).sub_(self.image_mean)\
                                         .div_(self.image_std)
                        else:
                            next_input = next_input.to(self.dtype)

                    next_input_dict[k] = next_input

            if not first:
                yield SimpleNamespace(**input_dict)
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input_dict = next_input_dict.copy()

        yield SimpleNamespace(**input_dict)

    def __len__(self):
        return len(self.dataloader)


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



