from __future__ import annotations

import os

from math import ceil, floor
import copy
import time

from collections import OrderedDict

import torch
from torch.utils.data import DataLoader, Sampler
from torchvision.datasets import ImageFolder

from types import SimpleNamespace
from typing import Optional, Union, Callable, Tuple, Sequence


class AdvanceImageFolder(ImageFolder):

    def __init__(self, 
                 root: str,
                 name: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 load_percentage: Optional[float] = None,
                 indices: Optional[list[int]] = None,
                 seed: Optional[int] = 12345):

        """
            Initialization of an AdvancedImageFolder dataset. It is possible do it into
            two different ways:

            - uning a root path, as common in ImageFolder datasets.
            - using another AdvancedImageFolder dataset. In this way it is possible to
              use images already loaded into memory and build efficiently subsets.

            Args:
                root (str): the path to root directory (as ImageFolder).
                name (str): the name of the dataset. If None, the name of the root will
                be used.
                transform (Callable): a function to transform the PIL images 
                (once loaded from memory).
                target_transform (Callable): a function to transform labels.
                load_percentage (float): the percentage of the dataset to consider.  
                Random samples will be taken without repetitions.
                indices (list of int): a list of indices to directly specify a subset of
                the dataset.
                seed (int): an integer for the RNG (for reproducibility).

            Raises:
                ValueError if:
                    -load_percentage is not in [0, 1].
                    -both indices and load_percentage are specified.
        """

        # check load_percentage 
        if load_percentage and not (0 <= load_percentage <= 1):
            raise ValueError(f"load_percentage = {load_percentage} not in interval \
                               [0, 1]")

        # check indices and load_percentage
        if (load_percentage is not None) and (indices is not None):
            raise ValueError("Zero or one of [load_percentage, indices] should be set. \
                              Found two.")

        self.load_percentage = load_percentage
        self.seed = seed

        # if name is None, use the root directory name as dataset name
        
        if not name:
            name = os.path.basename(os.path.dirname(os.path.join(root, "")))

        self.name = name

        # initialize from root
        super().__init__(root=root, 
                         transform=transform, 
                         target_transform=target_transform)

        if self.load_percentage:

            # total number of images
            n_images_tot = len(self.samples)

            # number of images to load
            n_images = int(ceil(n_images_tot * self.load_percentage))

            # get n_images random indices.
            generator = torch.Generator().manual_seed(seed)
            indices = torch.randperm(n_images_tot, 
                      generator=generator).tolist()[0: n_images]
        
        if indices is not None:
            self.samples = [self.samples[i] for i in indices]
        
        # ids lookup table: (id: index). Useful when subsetting and/or splitting. Indices
        # are always kept between 0 to len-1, while ids reference to original index.
        # Initialized with ids=indices.
        self.ids2indices = OrderedDict([(i, i) for i in range(len(self.samples))])
        self.ids = list(self.ids2indices.keys())

        self.pseudolabels = -torch.ones(len(self.samples)).long()

        # variables for dataset loaded into RAM.
        self.ram_samples = None


    def __repr__(self) -> str:
        
        return f"AdvanceImageFolder: [Name='{self.name}', Length={len(self)}, " + \
               f"Classes={len(self.classes)}, Root='{self.root}']"


    def subset(self, 
               indices: Optional[list[int]] = None,
               subset_name: Optional[str] = None,
               ids: Optional[list[int]] = None) -> AdvanceImageFolder:
        """
            Get a subset of current dataset. Use a list of integer indices to select 
            the samples of the subset or a list of interged images ids. 
           
            Note:
                every image has a unique index that specify its position in the sequence.
                every image has also a unique id number (may be equal to the index or 
                not).

            Args:
                indices (list of int): the indices to select.
                subset_name (str): the name of the split.
                ids (list of int): the ids to select.

            Raises:
                ValueError if zero or two of [indices, ids] are set.

            Returns:
                a subset of current dataset.
        """

        if not (indices is None) ^ (ids is None):
            raise ValueError("One and just one of [indices, ids] should be set.")

        subset_dataset = copy.copy(self)

        # if ids are specified -> select indices of given ids.
        if ids is not None:
            indices = [self.ids2indices[i] for i in ids]

        # select samples
        subset_dataset.samples = [subset_dataset.samples[i] for i in indices]

        # select pseudolabels
        subset_dataset.pseudolabels = torch.LongTensor([subset_dataset.pseudolabels[i] 
                                                       for i in indices])

        # update ids and ids2indices (id:index) lookup table                                    
        subset_dataset.ids = [subset_dataset.ids[i] for i in indices]
        subset_dataset.ids2indices = \
        OrderedDict([(subset_dataset.ids[i], i) for i in range(len(indices))])
        
        # select ram samples
        if subset_dataset.ram_samples is not None:
            subset_dataset.ram_samples = [subset_dataset.ram_samples[i] for i in indices]

        subset_name = f"({subset_name})" if subset_name is not None else '(subset)'
        subset_dataset.name = self.name + subset_name
        return subset_dataset


    @staticmethod
    def _collate_fn(batch: list[SimpleNamespace]) -> SimpleNamespace:
        """
            Collate function for batching SimpleNamespaces.
        """

        if not isinstance(batch[0], SimpleNamespace):
            raise ValueError("Not supported format, just SimpleNamespace!")

        # SimpleNamespace with fields:
        # image: PIL or Tensor
        # label: int
        # pseudolabel: int
        # index: int

        image       = [element.image for element in batch]
        label       = [element.label for element in batch]
        pseudolabel = [element.pseudolabel for element in batch]
        id          = [element.id for element in batch]
        index       = [element.index for element in batch]

        if isinstance(batch[0].image, torch.Tensor):
            out = None

            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum(x.numel() for x in image)
                storage = image[0].storage()._new_shared(numel)
                out = image[0].image.new(storage)

            image = torch.stack(image, 0, out=out)
    
        label = torch.LongTensor(label)
        pseudolabel = torch.LongTensor(pseudolabel)
        id = torch.LongTensor(id)
        index = torch.LongTensor(index)

        return SimpleNamespace(image=image, 
                               label=label,
                               pseudolabel=pseudolabel,
                               id=id,
                               index=index)


    def dataloader(self,
               batch_size: Optional[int] = 1,
               shuffle: bool = False, 
               sampler: Optional[Sampler] = None,
               batch_sampler: Optional[Sampler[Sequence]] = None,
               num_workers: int = 0, 
               pin_memory: bool = False, 
               drop_last: bool = False) -> DataLoader:

        """
            Get a torch.utils.data.DataLoader object associated with this dataset.

            Args:
                batch_size (int): the batch size.
                shuffle (bool): True to shuffle indices.
                sampler (Sampler): a sampler for indices.
                batch_sampler (Sampler): a sampler for batches.
                num_workers (int): the number of concurrent workers.
                pin_memory (bool): True to pin gpu memory.
                drop_last (bool): True to drop last batch if incomplete.

            Returns:
                a torch.utils.data.DataLoader
        """

        return DataLoader(dataset=self,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          sampler=sampler,
                          batch_sampler=batch_sampler,
                          num_workers=num_workers,
                          pin_memory=pin_memory,
                          drop_last=drop_last,
                          collate_fn=AdvanceImageFolder._collate_fn)

    def random_split(self, 
                     percentages: list[float],
                     split_names: Optional[list[str]] = None,
                     seed: Optional[int] = 1234) \
                     -> Union[Tuple[AdvanceImageFolder], AdvanceImageFolder]:
        """
            Split the dataset in subdatasets and return them as a tuple.

            Args:
                percentages (list of floats): a percentage for each partition.
                split_names (list of str): the names of splits.
                seed (int): seed for RNG.

            Raises:
                ValueError if some percentages are not in range [0, 1] or if the sum 
                is greater than zero.
            
            Returns:
                A tuple of AdvanceImageFolder with all splits. If just one split is asked
                than an AdvanceImageFolder is returned.
        """

        # check percentages
        for p in percentages:
            if not 0 <= p <= 1:
                raise ValueError("Percentages should be in interval [0, 1].")

        if sum(percentages) > 1:
            raise ValueError("Percentages sum should be less or equal to one.")

        if split_names is None or not isinstance(split_names, list):
            split_names = [None for _ in percentages]

        # convert percentages to lengths
        lengths = [floor(self.__len__() * p) for p in percentages]

        # get random indices
        generator = torch.Generator().manual_seed(seed)
        all_indices = torch.randperm(self.__len__(), 
                               generator=generator).tolist()[0: sum(lengths)]

        # for each subset save the random indices inside a list
        subsets_indices = [[] for _ in lengths]

        current_idx = 0
        for i, length in enumerate(lengths):
            for _ in range(length):
                subsets_indices[i].append(all_indices[current_idx])
                current_idx += 1
        
        # just one split
        if len(subsets_indices) == 1:
            return self.subset(subsets_indices[0], split_names[0])

        # more splits
        return tuple(self.subset(indices, split_names[i]) 
                     for i, indices in enumerate(subsets_indices))


    def update_pseudolabels(self,
                            values: Union[torch.LongTensor, list[int]],
                            indices: Optional[Union[torch.LongTensor, list[int]]] = None,
                            ids: Optional[Union[torch.LongTensor, list[int]]] = None):
        """
            Updates the pseudolabels with values at given indices or ids.

            Args:
                values (torch.LongTensor | list of int): the values of new pseudolabels.
                indices (torch.LongTensor | list of int): the indices of new 
                pseudolabels.
                indices (torch.LongTensor | list of int): the ids of new pseudolabels.
            
            Raises:
                ValueError if zero or both [indices, ids] are set.
        """ 

        if not (indices is None) ^ (ids is None):
            raise ValueError("One and just one of [indices, ids] should be set.")

        if ids is not None:
            indices = [self.ids2indices[i] for i in ids]

        if isinstance(values, list):
            values = torch.LongTensor(values)

        self.pseudolabels[indices] = values


    def loaded(self) -> bool:
        """
            Check if the dataset is currently loaded into RAM.

            Returns:
                True if the dataset is loaded into RAM, False otherwise.
        """

        return self.ram_samples is not None


    def clean_ram(self):
        """
            Cleans the dataset from RAM removing any alloacated variable.
        """
        self.ram_samples = None


    def load_ram(self,
                 num_workers: Optional[int] = 1,
                 batch_size: Optional[int] = 20,
                 verbose: Optional[bool] = False):
        """
            Load the dataset (or a part of it) into RAM to speed up memory accesses. 
            If a part of dataset is loaded, just that part will be returned, while the 
            part not loaded will be completely be ignored.

            Args:
                num_workers (int): the number of workers to use.
                batch_size (int): the batch_size to use for the loading.
                verbose (bool): True to print debug logs.
        """  

        # disabling transforms
        transform = self.transform
        target_transform = self.target_transform
        self.transform = None
        self.target_transform = None


        # set the getitem function to read images from disk.
        self.ram_samples = None

        if verbose:
            print("Loading dataset to ram...")
            start = time.time()

        # take a dataloder for multi worker processing
        loader = DataLoader(self, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, drop_last=False, 
                            collate_fn=AdvanceImageFolder._collate_fn)

        ram_samples = []

        for batch in loader:
            ram_samples = ram_samples + list(zip(batch.image, batch.label))

        # set the getitem function to read images from ram.
        self.ram_samples = ram_samples

        # re-enabling transforms
        self.transform = transform
        self.target_transform = target_transform

        if verbose:
            end = time.time()
            print(f"Time spent: {end-start:6.4}")
            print()


    def __getitem__(self, index: int) -> SimpleNamespace:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        if not self.loaded():
            path, label = self.samples[index]
            image = self.loader(path)
        else:
            image, label = self.ram_samples[index]

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return SimpleNamespace(image=image, 
                               label=label,
                               pseudolabel=self.pseudolabels[index],
                               id=self.ids[index],
                               index=index)


    def set_transform(self, transform):
        """
            Change the transform to apply to inputs.

            Args:
                transform (torchvision.transform): the transform to apply to inputs.
        """
        self.transform = transform

        
    def set_target_transform(self, target_transform):
        """
            Change the transform to apply to labels.

            Args:
                transform (torchvision.transform): the transform to apply to labels.
            
        """
        self.target_transform = target_transform


    def n_classes(self):
        """
            Returns the number of classes of the current dataset.
        """
        return len(self.classes)
