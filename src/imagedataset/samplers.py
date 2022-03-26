from torch.utils.data import Sampler, Dataset
import torch
import torch.distributed as dist
from typing import Iterable, Optional, List
from math import ceil


class PartitionDistributedSampler(Sampler):
    """
        A distributed sampler that does not add new samples to the dataset and that 
        divides all the indices into chunks (mutually exclusive) and gives them to 
        different ranks.


        For example (3 ranks)

        ALL INDICES = [0 1 2 3 4 5 6]

        CHUNKS 
                -rank 0 = [0 1 2]
                -rank 1 = [3 4 5] 
                -rank 2 = [6]

        NOTE:
            the last rank can have less indices than others.
    """

    def __init__(self, 
                 dataset: Dataset, 
                 world_size: Optional[int] = None, 
                 rank: Optional[int] = None,
                 shuffle: Optional[bool] = False,
                 seed: Optional[int] = 123):
        """
            Initializes the sampler.

            Args:
                dataset (Dataset): the dataset.
                world_size (int, optional): the world size.
                rank (int, optional): the current rank.
        """

        # get world size and rank if they are None.
        if world_size is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            world_size = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.dataset    = dataset
        self.world_size = world_size
        self.rank       = rank
        self.seed       = seed
        self.shuffle    = shuffle
        self.set_epoch(0)

    def set_epoch(self, epoch: int):

        self.epoch = epoch

        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            self.all_indices = torch.randperm(len(self.dataset), generator=generator).tolist() 
        else:
            self.all_indices = list(range(len(self.dataset)))

        # number of samples for each rank
        self.samples_per_rank = ceil(len(self.dataset)/self.world_size)

        self.chunks = [self.all_indices[r * self.samples_per_rank : 
                       (r+1) * self.samples_per_rank] for r in range(self.world_size)]

        self.indices = self.chunks[self.rank]


    def get_partitions(self) -> List[List[int]]:
        """ Get all the partitions."""
        return self.chunks


    def __iter__(self) -> Iterable:
        return iter(self.indices)


    def __len__(self) -> int:
        return len(self.indices)