# Inspired to the APEX example:
# https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py#L256
# and to TIMM loader:
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/loader.py

from torch.utils.data import DataLoader
import torch
from types import SimpleNamespace
import numpy as np
from typing import Callable, List
import random


def fast_collate(batch: List[SimpleNamespace], memory_format) -> SimpleNamespace:

    images = [sample.image for sample in batch]

    # collate targets

    # case classification -> ints
    if isinstance(batch[0].label, int):
        targets = torch.tensor([sample.label for sample in batch], dtype=torch.int64)
    
    # case segmentation -> imgs (np.ndarray)
    elif isinstance(batch[0].label, np.ndarray):
        targets = [sample.label for sample in batch]
        w = targets[0].size[0]
        h = targets[0].size[1]
        c = targets[0].size[2]

        targets_tensor = torch.zeros((len(images), c, h, w), dtype=torch.uint8) \
                         .contiguous(memory_format=memory_format)
                         
        for i, target in enumerate(targets):
            array = np.asarray(target, dtype=np.uint8)
            if array.ndim < 3:
                array = np.expand_dims(array, axis=-1)
            array = np.rollaxis(array, 2)
            targets_tensor[i] += torch.from_numpy(array)
        
        targets = targets_tensor

    elif isinstance(batch[0].label, tuple):
        msg = f"Collate of Tuple targets not implemented yet!"
        raise NotImplementedError(msg)
    else:
        msg = f"Collate of targets {type(batch[0].label)} not implemented!"
        raise NotImplementedError(msg)

    # collate images

    w = images[0].size[0]
    h = images[0].size[1]
    c = images[0].size[2]

    images_tensor = torch.zeros((len(images), c, h, w), dtype=torch.uint8) \
                         .contiguous(memory_format=memory_format)

    for i, img in enumerate(images):
        array = np.asarray(img, dtype=np.uint8)
        if array.ndim < 3:
            array = np.expand_dims(array, axis=-1)
        array = np.rollaxis(array, 2)
        images_tensor[i] += torch.from_numpy(array)

    return SimpleNamespace(image=images_tensor, label=targets)



class CudaLoader:

    def __init__(self,
                 loader: DataLoader,
                 mean: tuple,
                 std: tuple,
                 fp16=False):

        self.dataloader = loader
        self.mean = torch.tensor([x * 255 for x in mean]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([x * 255 for x in std]).cuda().view(1, 3, 1, 1)
        self.fp16 = fp16

        if fp16:
            self.mean = self.mean.half()
            self.std = self.std.half()

    def __iter__(self):

        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in self.loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)
                if self.fp16:
                    next_input = next_input.half().sub_(self.mean).div_(self.std)
                else:
                    next_input = next_input.float().sub_(self.mean).div_(self.std)

            if not first:
                yield input, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target

        yield input, target

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



