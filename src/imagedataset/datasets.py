from __future__ import annotations
import warnings

from types import SimpleNamespace
from typing import Optional, Union,  Tuple, Sequence, List

import os
import copy
import pickle
from math import ceil, floor

from collections import OrderedDict

import numpy as np
import cv2

import torch
from torch.utils.data import DataLoader, Sampler, Dataset

from imagedataset.enums import OutputFormat

from .utils import (
    get_dir_names, 
    get_file_names, 
    match_file_names, 
    array_to_bytes, 
    bytes_to_array
)

from .dataloaders import CpuLoader, CudaLoader, get_collate
from .operations import  ImageLoader, FileLoader
from .imageloaders import LoaderPIL
from enum import Enum


def from_database(path, 
                  image_size: Optional[Tuple[int,int]] = None, 
                  load_labels: Optional[bool] = True) \
                      -> Union[BasicImageFolder, AdvancedImageFolder]:
    """
        NOTE: requires `lmdb` installed
        Get a BasicImageFolder or an AdvanceImageFolder from a LMDB database file.

        Args:
            path (str): the path to the LMDB.
            image_size (Tuple, optional): A new image size for database images, None to
            keep original image size.
            load_labels (bool, optional): True to load labels to RAM, False to keep 
            labels inside the database.
    """

    import lmdb

    env = lmdb.open(path, readonly=True, lock=True)
    txn = env.begin(write=False)

    dataset = pickle.loads(txn.get('__object__'.encode()))
    dataset.loaded_labels = None

    if load_labels:
        labels = []
        for i in range(len(dataset)):
            original_index = dataset.ids[i]
            label = int(bytes_to_array(txn.get(f"label{original_index}".encode())))
            labels.append(label)  
        dataset.loaded_labels = labels

    dataset._initLMDB_env(path)

    if image_size is not None:
        dataset.image_size = image_size

    return dataset


class DatasetState(Enum):
    DATABASE_NOT_AVAILABLE  = 0
    DATABASE_NUMPY          = 1
    DATABASE_JPG            = 2
    RAM_NOT_AVAILABLE       = 3
    RAM_NUMPY               = 4
    RAM_JPG                 = 5


class BasicImageFolder(Dataset):

    """
        A simple dataset of images. It should be structured as (names are arbitrary):
    
          root/
            ├── class 1 
            │     ├── image_1.ext
            │     ├── image_2.ext
            │     ├── image_3.ext
            │     ├── ...
            │     └── image_i.ext
            │
            ├── ...
            │
            └── class_k
                  ├── image_1.ext
                  ├── image_2.ext
                  ├── image_3.ext
                  ├── ...
                  └── image_j.ext

        NOTE:
            - for classification the labels are taken fron the names of the folders.
            = additionally, another root path can be passes with a file loader to load 
              custom labels (for segmentation, object detection, etc).

        Args:
            root (str): the root path to images.
            label_root (str, optional): the root to labels (if None, the names of folders
            will be uses as labels for classification).
            image_loader(ImageLoader, optional): an ImageLoader to load images from paths.
            label_loader(FileLoader, optional): a generic FileLoader to load labels from
            paths.
            load_percentage(float, optional): a float in (0, 1]. Just this percentage of
            the dataset will be considere (samples will be selected randomly).
            keep_indices(list[int], optional): the indices to consider (if None all 
            dataset will be considered).
            filter_image_formats (list[str], optional): the image formats to consider,
            other formats will be ignored.
            filter_label_formats (list[str], optional): the label formats to consider,
            other formats will be ignored.
            dataset_name (str, optional): the name of this dataset.
            seed (int, optional): the seed for rng.
    """

    def __init__(self, 
                 root: str,
                 label_root: Optional[str] = None,
                 image_loader: Optional[ImageLoader] = LoaderPIL(),
                 label_loader: Optional[FileLoader] = None,
                 load_percentage: Optional[float] = None,
                 keep_indices: Optional[List[int]] = None,
                 filter_image_formats: Optional[list[str]] = ["png", "jpg", "jpeg"],
                 filter_label_formats: Optional[list[str]] = None,
                 dataset_name: Optional[str] = None,
                 seed: Optional[int] = 1234):

        self.database_state = DatasetState.DATABASE_NOT_AVAILABLE
        self.ram_state      = DatasetState.RAM_NOT_AVAILABLE

        # Current image size of the dataset.
        self._image_size         = image_loader.size
        self.ram_image_size      = None
        self.database_image_size = None
    
        # True if we have external labels, False if the labels are the names of folders.
        self.external_labels = label_root is not None


        # check load_percentage 
        if load_percentage and not (0 <= load_percentage <= 1):
            raise ValueError(f"load_percentage = {load_percentage} not in interval " +
                              "[0, 1]")

        # check indices and load_percentage
        if (load_percentage is not None) and (keep_indices is not None):
            raise ValueError("Zero or one of [load_percentage, keep_indices] should " +
                             "be set. Found two.")

        # check that we have a loader for external labels.
        if (label_loader is None) and self.external_labels:
            raise ValueError("'label_root' selected but 'label_loader' is None.")

        # roots
        self.images_root = root
        self.labels_root = label_root

        # dataset name
        if not dataset_name:
            self.dataset_name = os.path.basename(os.path.dirname(os.path.join(root, "")))
        else:
            self.dataset_name = dataset_name

        # formats allowed
        self.image_formats  = set(filter_image_formats)  if filter_image_formats \
                                                         else None
        self.label_formats = set(filter_label_formats) if filter_label_formats \
                                                         else None

        # loaders and decoders
        self.image_loader  = image_loader
        self.label_loader  = label_loader
        self.image_decoder = image_loader.decoder(keep_resizer=False)
        self.image_resizer = None
        

        # SETUP DATASET
        self.classes       = None  # class names list
        self.class2index   = None  # dict {class_name:index}
        self.images        = None  # list of the paths to images
        self.labels        = None  # list of the paths to labels
        self.loaded_images = None  # list of images (loaded)
        self.loaded_labels = None  # list of labels (loaded)


        # fill the fields above...
        self._setup_dataset()

        # IDS

        # ids lookup table: (id: index). Useful when subsetting and/or splitting. Indices
        # are always kept between 0 to len-1, while ids reference to original index.
        # Initialized with ids=indices.
        self.ids2indices = OrderedDict([(i, i) for i in range(len(self.images))])
        self.ids = list(self.ids2indices.keys())


        # SUBSETTING THE DATASET IF NEEDED
        self.load_percentage = load_percentage
        self.seed = seed


        if self.load_percentage:

            # total number of images
            n_images_tot = len(self.images)

            # number of images to load
            n_images = int(ceil(n_images_tot * self.load_percentage))

            # get n_images random indices.
            generator = torch.Generator().manual_seed(seed)
            keep_indices = torch.randperm(n_images_tot, 
                           generator=generator).tolist()[0: n_images]
        
        if keep_indices is not None:
            if self.images is not None:
                self.images = [self.images[i] for i in keep_indices]
            if self.labels is not None:
                self.labels = [self.labels[i] for i in keep_indices]
            if self.loaded_images is not None:
                self.loaded_images = [self.loaded_images[i] for i in keep_indices]
            if self.loaded_labels is not None:
                self.loaded_labels = [self.loaded_labels[i] for i in keep_indices]


    def get_state(self):
        return self.ram_state.name, self.database_state.name


    @property
    def image_size(self) -> Tuple[int, int]:
        """ Get the current image size. """
        return self._image_size
    
    
    @image_size.setter
    def image_size(self, image_size: Tuple[int, int]):
        """ Set a new image size. """
        

        if (self.ram_state != DatasetState.RAM_NOT_AVAILABLE) and\
            (image_size != self.ram_image_size) and image_size is not None:
   
            from_size = self.ram_image_size if self.ram_image_size is not None else\
                        "(*, *)"
            msg = f"RAM images will be resized from {from_size} to {image_size}!"
            warnings.warn(msg)

        elif (self.database_state != DatasetState.DATABASE_NOT_AVAILABLE) and\
            (image_size != self.database_image_size) and image_size is not None:

            from_size = self.database_image_size if self.database_image_size is not None\
                                                 else "(*, *)"
            msg = f"Database images will be resized from {from_size} to {image_size}!"
            warnings.warn(msg)
        
        # deepcopy loaders to set new image size (since subsetting/split makes just
        # shallow copies and it is possible that the loaders are shared among many
        # instances of dataset).
        self.image_loader  = copy.deepcopy(self.image_loader)
        self.image_loader.size  = image_size # set the new image_size

        # create the new decoder and resizer
        self.image_decoder = self.image_loader.decoder(keep_resizer=True)

        if self.ram_state == DatasetState.RAM_JPG:
            # images in RAM as JPG -> use decoder
            if image_size != self.ram_image_size:
                self.image_decoder.size = image_size
            else:
                self.image_decoder.size = None

        elif self.ram_state == DatasetState.RAM_NUMPY:
            # images in RAM as NUMPY -> use resizer
            self.image_resizer = self.image_loader.resizer()
            if image_size != self.ram_image_size:
                self.image_resizer.size = image_size
            else:
                self.image_resizer.size = None

        elif self.database_state == DatasetState.DATABASE_JPG:
            # images in DB as JPG -> use decoder
            if image_size != self.database_image_size:
                self.image_decoder.size = image_size
            else:
                self.image_decoder.size = None

        elif self.database_state == DatasetState.DATABASE_NUMPY:
            # images in DS as NUMPY -> use resizer
            self.image_resizer = self.image_loader.resizer()
            if image_size != self.database_image_size:
                self.image_resizer.size = image_size
            else:
                self.image_resizer.size = None

        self._image_size = image_size


    def set_loader(self, loader: ImageLoader):
        """Set the loader."""
        self.image_loader = loader
        self.image_size = loader.size


    def subset(self, 
               indices: Optional[list[int]] = None,
               subset_name: Optional[str] = None,
               ids: Optional[list[int]] = None) -> BasicImageFolder:
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

        # select images
        if self.images is not None:
            subset_dataset.images = [subset_dataset.images[i] for i in indices]
        
        # select labels
        if self.labels is not None:
            subset_dataset.labels = [subset_dataset.labels[i] for i in indices]

        # select loaded images
        if self.loaded_images is not None:
            subset_dataset.loaded_images = [subset_dataset.loaded_images[i] 
                                            for i in indices]
        # select loaded labels
        if self.loaded_labels is not None:
            subset_dataset.loaded_labels = [subset_dataset.loaded_labels[i] 
                                            for i in indices]


        # update ids and ids2indices (id:index) lookup table                                    
        subset_dataset.ids = [subset_dataset.ids[i] for i in indices]
        subset_dataset.ids2indices = \
        OrderedDict([(subset_dataset.ids[i], i) for i in range(len(indices))])
        

        subset_name = f"({subset_name})" if subset_name is not None else '(subset)'
        subset_dataset.name = self.dataset_name + subset_name

        return subset_dataset


    def random_split(self, 
                     percentages: list[float],
                     split_names: Optional[list[str]] = None,
                     seed: Optional[int] = 1234) \
                     -> Union[Tuple[BasicImageFolder], BasicImageFolder]:
        """
            Splits the dataset in subdatasets and return them as a tuple.

            Args:
                percentages (list of floats): a percentage for each partition.
                split_names (list of str): the names of splits.
                seed (int): seed for RNG.

            Raises:
                ValueError if some percentages are not in range [0, 1] or if the sum 
                is greater than zero.
            
            Returns:
                A tuple of BasicImageFolder with all splits. If just one split is asked
                than an BasicImageFolder is returned.
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

    def __repr__(self):
        """ Returns the repr of this object. """

        repr = f"<{self.__class__.__name__}[{self.dataset_name}, {len(self)}] "

        if self.ram_state == DatasetState.RAM_NUMPY:
            source = "RAM_NP"
            image_size = str(self.ram_image_size) if self.ram_image_size is not None else\
                "(*, *)"
            reader = str(self.image_resizer)
        
        elif self.ram_state == DatasetState.RAM_JPG:
            source = "RAM_JPG"
            image_size = str(self.ram_image_size) if self.ram_image_size is not None else\
                "(*, *)"
            reader = str(self.image_decoder)

        elif self.database_state == DatasetState.DATABASE_NUMPY:
            source = "DB_NP"
            image_size = str(self.database_image_size) if self.database_image_size\
                        is not None else "(*, *)"
            reader = str(self.image_resizer)

        elif self.database_state == DatasetState.DATABASE_JPG:
            source = "DB_JPG"
            image_size = str(self.database_image_size) if self.database_image_size\
                        is not None else "(*, *)"
            reader = str(self.image_decoder)
            loader, decoder = "", ""
        else:
            source = "DISK"
            image_size = "(*, *)"
            reader = str(self.image_loader)

        repr += f"{source}{image_size} -> {reader}>"
        return repr


    def __len__(self) -> int:
        """ Returns the length of the dataset. """
        return len(self.images)


    def n_classes(self) -> int:
        """ Returns the number of classes in the dataset. """
        return len(self.classes)


    def _setup_dataset(self):
        """ Sets up the dataset. """

        self._get_classes()

        if self.external_labels:
            self._get_images_and_labels()
        else:
            self._get_images()


    def _check_image(self, path: str) -> bool:
        """ Returns True if path has a valid format for images. """
        if self.image_formats is None:
            return True

        _, file_extension = os.path.splitext(path)
        file_extension = file_extension[1:] # remove . from extension

        return file_extension in self.image_formats


    def _get_classes(self):
        """ Setups of classes and class2idx. """
        self.classes = sorted(get_dir_names(self.images_root))
        self.class2index = {cl : idx for idx, cl in enumerate(self.classes)}


    def _get_images(self):
        """ Setups images (for image classification). """
        self.images = []
        self.loaded_labels = []

        for class_name, class_idx in self.class2index.items():
            
            # path to current class for images and targets
            class_image_path  = os.path.join(self.images_root, class_name)

            for image in sorted(get_file_names(class_image_path)):  

                image_path = os.path.join(class_image_path, image)

                if self._check_image(image_path):
                    self.images.append(image_path)
                    self.loaded_labels.append(class_idx)


    def _get_images_and_labels(self, warn_mismatch: Optional[bool] = True):
        """ Setups the dataset (get labels (external) and images). """

        self.images = []
        self.labels = []

        for class_name in self.class2index.keys():
            
            # path to current class for images and targets
            class_image_path  = os.path.join(self.images_root, class_name)
            class_target_path = os.path.join(self.labels_root, class_name)

            matching_files = match_file_names(path1=class_image_path,
                                              path2=class_target_path, 
                                              formats_1=self.image_formats, 
                                              formats_2=self.label_formats, 
                                              warn_mismatch=warn_mismatch)

            for image, target in matching_files:

                image_path = os.path.join(class_image_path, image)
                target_path = os.path.join(class_target_path, target)
                
                self.images.append(image_path)
                self.labels.append(target_path)
    

    def __getitem__(self, index) -> SimpleNamespace:
        """ 
            Get the item at `index`.

            Two modes: if the dataset is from a database or not.
            
            In database mode images and labels can be retrieved from the database
            or from the RAM (if they have been loaded). RAM has priority over the 
            database.

            In non-database mode images and labels can be retrived (as files) in the disk
            or from RAM (if they have been loaded). RAM has priority over the disk.

            In addition images can be loaded both from numpy array or from image files.
            -numpy        = MORE MEMORY, LESS COMPUTATIONS
            -image files  = LESS MEMORY, MORE COMPUTATIONS

        """

        # LOAD IMAGE

        # FROM RAM
        if self.ram_state == DatasetState.RAM_JPG:
            image = self.image_decoder(self.loaded_images[index])
        elif self.ram_state == DatasetState.RAM_NUMPY:
            image = self.loaded_images[index]
            if self.image_resizer is not None:
                image = self.image_resizer(image)

        # FROM DB
        elif self.database_state == DatasetState.DATABASE_JPG:
            if self.txn is None:  
                self._initLMDB_txn()
            original_index = self.ids[index]
            image = self.image_decoder(self.txn.get(f"image{original_index}".encode()))

        elif self.database_state == DatasetState.DATABASE_NUMPY:
            if self.txn is None:  
                self._initLMDB_txn()
            original_index = self.ids[index]
            image = bytes_to_array(self.txn.get(f"image{original_index}".encode()))
            if self.image_resizer is not None:
                image = self.image_resizer(image)

        # FROM DISK
        else:
            image = self.image_loader(self.images[index])

        # LOAD LABEL
        if self.loaded_labels is not None:
            label = self.loaded_labels[index]
        elif self.database_state != DatasetState.DATABASE_NOT_AVAILABLE:
            original_index = self.ids[index]
            label = bytes_to_array(self.txn.get(f"label{original_index}".encode()))
        else:
            label = self.label_loader(self.labels[index])
        
        return SimpleNamespace(image=image, label=label)

        # # MODE: NOT DATABASE
        # if not self.from_database:

        #     # IMAGES 
        #     if self.loaded_images is not None:                  # 1) NUMPY IMAGES (RAM)
        #         if self.images_loaded_as_numpy:
        #             image = self.loaded_images[index]
        #         else:                                           # 2) BYTES IMAGES (RAM)
        #             image = self.image_decoder(self.loaded_images[index])
        #     else:                                               # 3) IMAGES (DISK)
        #         image = self.image_loader(self.images[index])

        #     # LABELS
        #     if self.loaded_labels is not None:                  # 1) LABELS (RAM)
        #         label = self.loaded_labels[index]
        #     else:                                               # 2) LABELS (DISK)
        #         label = self.label_loader(self.labels[index])


        # # MODE: DATABASE
        # else:

        #     if self.txn is None:  
        #         self._initLMDB_txn()

        #     # IMAGE
        #     if self.loaded_images is not None:
        #         if self.images_loaded_as_numpy:              # 1) NUMPY IMAGES (RAM)
        #             image = self.loaded_images[index]
        #         else:                                        # 2) BYTES IMAGES (RAM)
        #             image = self.image_decoder(self.loaded_images[index])
        #     else:
        #         if self.images_database_as_numpy:            # 3) NUMPY IMAGES (DATABASE)
        #             image = bytes_to_array(self.txn.get(f"image{index}".encode()))
        #         else:                                        # 4) BYTES IMAGES (DATABASE)
        #             image = self.image_decoder(self.txn.get(f"image{index}".encode()))

        #     # LABEL
        #     if self.loaded_labels is not None:               # 1) LABELS (RAM)
        #         label = self.loaded_labels[index]
        #     else:                                            # 2) LABELS (DATABASE)
        #         label = bytes_to_array(self.txn.get(f"label{index}".encode()))

        # return SimpleNamespace(image=image, label=label)


    def load_ram(self,
                 numpy_images: Optional[bool] = False,
                 num_workers: Optional[int]   = 4) -> BasicImageFolder:
        """ 
            Loads to RAM the entire dataset.

            Args:
                numpy_images (bool, optioanl): True to load images as np.ndarray in 
                memory (it will speed up loading but it consumes more memory).
                num_workers (int, optional): The number of workers for the loading.
            Returns:
                self
        """

        # new lables/images loaded in memory (in lists).
        loaded_images = []
        loaded_labels = []

        # get a collate function that does not alter data and do not convert to tensor.
        if self.external_labels:
            label_format = OutputFormat.UNALTERED_UNIT8_NUMPY
        else:
            label_format = OutputFormat.UNALTERED_INT64_NUMPY

        collate_fn = get_collate(image_format=OutputFormat.UNALTERED_UNIT8_NUMPY,
                                 label_format=label_format,
                                 other_fields_format=OutputFormat.UNALTERED_INT64_NUMPY,
                                 memory_format=torch.contiguous_format)

        dataloader = DataLoader(self, 
                                batch_size=1, 
                                num_workers=num_workers, 
                                shuffle=False, 
                                drop_last=False,
                                collate_fn=collate_fn)
        
        for batch in dataloader:

            image = batch.image[0]
            if not numpy_images:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image = cv2.imencode(".jpg", image)[1].tobytes()
            loaded_images.append(image)
            loaded_labels.append(batch.label[0])

        # set images
        self.loaded_images   = loaded_images
        self.ram_image_size = self._image_size

        # set labels
        self.loaded_labels = loaded_labels

        # keep track if images have been loaded as numpy
        self.ram_state = DatasetState.RAM_NUMPY if numpy_images else DatasetState.RAM_JPG

        return self

    def cudaloader(self,
                   rank: Optional[int] = None,
                   image_format: Optional[OutputFormat] = OutputFormat.NCHW_FLOAT32_TENSOR,
                   label_format: Optional[OutputFormat] = OutputFormat.UNALTERED_INT64_TENSOR,
                   image_mean: Optional[tuple] = None,
                   image_std: Optional[tuple] = None,
                   memory_format: Optional[torch.memory_format] = torch.contiguous_format,
                   batch_size: Optional[int] = 1,
                   shuffle: bool = False, 
                   sampler: Optional[Sampler[int]] = None,
                   batch_sampler: Optional[Sampler[Sequence[int]]] = None,
                   num_workers: int = 0,
                   pin_memory: bool = False, 
                   drop_last: bool = False) -> CudaLoader:

        """ 
            NOTE:
                only images are converted in range [0, 1] and they should be floats.

            Get a torch.utils.data.DataLoader object associated with this dataset 
            wrapped inside a CudaLoader with prefatching and fast load to gpu.

            Args:
                rank (int, optional): the local rank (device).
                image_format (OutputFormat, optional): the image output format.
                label_format (OutputFormat, optional): the label output format.
                image_mean (tuple, optional): the mean to subtract to images.
                image_std (tuple, optional): the std to divide the images.
                memory_format (torch.memory_format, optional): the torch memory format
                for tensors.
                batch_size (int, optional): the batch size.
                shuffle (bool, optional): True to shuffle indices.
                sampler (Sampler, optional): a sampler for indices.
                batch_sampler (Sampler, optional): a sampler for batches.
                num_workers (int, optional): the number of concurrent workers.
                pin_memory (bool, optional): True to pin gpu memory.
                drop_last (bool, optional): True to drop last batch if incomplete.

             Returns: 
                a CudaLoader
        """

        assert "TENSOR" in image_format.name and "FLOAT" in image_format.name,\
        "image_format should be a TENSOR format of FLOAT"

        collate_fn = get_collate(image_format=image_format, 
                                 label_format=label_format,
                                 other_fields_format=OutputFormat.UNALTERED_INT64_TENSOR, 
                                 memory_format=memory_format)

        dataloader = DataLoader(self, 
                                batch_size=batch_size, 
                                shuffle=shuffle, 
                                sampler=sampler, 
                                batch_sampler=batch_sampler, 
                                num_workers=num_workers, 
                                pin_memory=pin_memory, 
                                drop_last=drop_last,
                                collate_fn=collate_fn)

        return CudaLoader(loader=dataloader, 
                          image_format=image_format, 
                          image_mean=image_mean, 
                          image_std=image_std, 
                          scale_image_floats=True, 
                          rank=rank)

    def cpuloader(self,
                  image_format: Optional[OutputFormat] = OutputFormat.NCHW_FLOAT32_TENSOR,
                  label_format: Optional[OutputFormat] = OutputFormat.UNALTERED_INT64_TENSOR,
                  image_mean: Optional[tuple] = None,
                  image_std: Optional[tuple] = None,
                  memory_format: Optional[torch.memory_format] = torch.contiguous_format,
                  batch_size: Optional[int] = 1,
                  shuffle: bool = False, 
                  sampler: Optional[Sampler[int]] = None,
                  batch_sampler: Optional[Sampler[Sequence[int]]] = None,
                  num_workers: int = 0,
                  pin_memory: bool = False, 
                  drop_last: bool = False) -> CudaLoader:

        """ 
            NOTE:
                only images are converted in range [0, 1] and they should be floats.

            Args:
                image_format (OutputFormat, optional): the image output format.
                label_format (OutputFormat, optional): the label output format.
                image_mean (tuple, optional): the mean to subtract to images.
                image_std (tuple, optional): the std to divide the images.
                memory_format (torch.memory_format, optional): the torch memory format
                for tensors.
                batch_size (int, optional): the batch size.
                shuffle (bool, optional): True to shuffle indices.
                sampler (Sampler, optional): a sampler for indices.
                batch_sampler (Sampler, optional): a sampler for batches.
                num_workers (int, optional): the number of concurrent workers.
                pin_memory (bool, optional): True to pin gpu memory.
                drop_last (bool, optional): True to drop last batch if incomplete.

             Returns: 
                a CpuLoader
        """

        assert "TENSOR" in image_format.name and "FLOAT" in image_format.name,\
        "image_format should be a TENSOR format of FLOAT"

        collate_fn = get_collate(image_format=image_format, 
                                 label_format=label_format,
                                 other_fields_format=OutputFormat.UNALTERED_INT64_TENSOR, 
                                 memory_format=memory_format)

        dataloader = DataLoader(self, 
                                batch_size=batch_size, 
                                shuffle=shuffle, 
                                sampler=sampler, 
                                batch_sampler=batch_sampler, 
                                num_workers=num_workers, 
                                pin_memory=pin_memory, 
                                drop_last=drop_last,
                                collate_fn=collate_fn)

        return CpuLoader(loader=dataloader, 
                          image_format=image_format, 
                          image_mean=image_mean, 
                          image_std=image_std, 
                          scale_image_floats=True)


    def write_database(self,
                       path: str,
                       numpy_images: Optional[bool] = False,
                       num_workers: Optional[int] = 4,
                       map_size: Optional[int] = int(1e14),
                       verbose: Optional[bool] = False):
        """ 
            NOTE: requires `lmdb` installed.

            Writes the entire dataset inside a LMDB database.

            Args:
                path (str): the path where to save the database.
                numpy_images (bool, optional): True to save images as numpy arrays in the
                database.
                num_workers (int, optional): the number of workers for the saving.
                map_size (int, optional): the max number of kB allowed for the database.
                verbose (bool, optional): True to print status.

        """
        import lmdb

        env = lmdb.open(path, map_size=map_size, lock=False)
        txn = env.begin(write=True)

        if self.external_labels:
            label_format = OutputFormat.UNALTERED_UNIT8_NUMPY
        else:
            label_format = OutputFormat.UNALTERED_INT64_NUMPY

        collate_fn = get_collate(image_format=OutputFormat.UNALTERED_UNIT8_NUMPY,
                                 label_format=label_format, 
                                 other_fields_format=OutputFormat.UNALTERED_INT64_NUMPY,
                                 memory_format=torch.contiguous_format)

        dataloader = DataLoader(self, 
                                batch_size=1, 
                                num_workers=num_workers, 
                                shuffle=False, 
                                drop_last=False,
                                collate_fn=collate_fn)

        print_step = len(dataloader)//10

        for idx, batch in enumerate(dataloader):
            image = batch.image[0]

            if verbose and idx % print_step == 0:
                print(f"Image {idx}/{len(dataloader)}")
                
            if numpy_images:
                image = array_to_bytes(image)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image = cv2.imencode(".jpg", image)[1].tobytes()

            label = array_to_bytes(batch.label[0])
            txn.put(('image'+ str(idx)).encode(), image)
            txn.put(('label'+ str(idx)).encode(), label)


        # in the save object the RAM images are not loaded because they are a Runtime
        # loaded objects, so we set them to None, we save the object and then we set
        # them back.

        # Save tmp references
        tmp_loaded_images = self.loaded_images
        tmp_loaded_labels = self.loaded_labels 
        tmp_ram_image_size = self.ram_image_size

        # remove references from current object
        self.loaded_images = None
        self.loaded_labels = None
        self.ram_image_size = None

        # update state of dataset
        self.database_state = DatasetState.DATABASE_NUMPY if numpy_images \
                              else DatasetState.DATABASE_JPG
        self.database_image_size = self._image_size
        self.image_resizer = self.image_loader.resizer()
        self.image_resizer.size = None
        
        # save the object
        encoded_object = pickle.dumps(self)
        txn.put('__object__'.encode(), encoded_object)

        # take references back.
        self.loaded_images = tmp_loaded_images
        self.loaded_labels = tmp_loaded_labels
        self.ram_image_size = tmp_ram_image_size

        txn.commit()
        env.close()

        # initialize database for current object
        self._initLMDB_env(path)


    def _initLMDB_env(self, path):
        """ Initializes the database from a path. """
        import lmdb
        self.lmdb_env = lmdb.open(path, readonly=True, lock=False)
        self.txn = None

    def _initLMDB_txn(self):
        """ Initializes the txn of current envirioment. """
        self.txn = self.lmdb_env.begin(write=False)


class AdvancedImageFolder(BasicImageFolder):
    """ 
        NOTE:
            it is very similar to BasicImageFolder.
            Additionally it is possible to save pseudolabels for image classification.
            The SimpleNamespace returned by __getitem__ contains:
            =image
            -label
            -id
            =index
            -pseudolabel

        Here the documentation of BasicImageFolder is repored.

        A simple dataset of images. It should be structured as (names are arbitrary):
    
          root/
            ├── class 1 
            │     ├── image_1.ext
            │     ├── image_2.ext
            │     ├── image_3.ext
            │     ├── ...
            │     └── image_i.ext
            │
            ├── ...
            │
            └── class_k
                  ├── image_1.ext
                  ├── image_2.ext
                  ├── image_3.ext
                  ├── ...
                  └── image_j.ext

        NOTE:
            - for classification the labels are taken fron the names of the folders.
            = additionally, another root path can be passes with a file loader to load 
              custom labels (for segmentation, object detection, etc).

        Args:
            root (str): the root path to images.
            label_root (str, optional): the root to labels (if None, the names of folders
            will be uses as labels for classification).
            image_loader(ImageLoader, optional): an ImageLoader to load images from paths.
            label_loader(FileLoader, optional): a generic FileLoader to load labels from
            paths.
            load_percentage(float, optional): a float in (0, 1]. Just this percentage of
            the dataset will be considere (samples will be selected randomly).
            keep_indices(list[int], optional): the indices to consider (if None all 
            dataset will be considered).
            filter_image_formats (list[str], optional): the image formats to consider,
            other formats will be ignored.
            filter_label_formats (list[str], optional): the label formats to consider,
            other formats will be ignored.
            dataset_name (str, optional): the name of this dataset.
            seed (int, optional): the seed for rng.
    """
    def __init__(self,
                 root: str,
                 label_root: Optional[str] = None,
                 image_loader: Optional[ImageLoader] = LoaderPIL(),
                 label_loader: Optional[FileLoader] = None,
                 load_percentage: Optional[float] = None,
                 keep_indices: Optional[List[int]] = None,
                 filter_image_formats: Optional[list[str]] = ["png", "jpg", "jpeg"],
                 filter_label_formats: Optional[list[str]] = None,
                 dataset_name: Optional[str] = None,
                 seed: Optional[int] = 1234):
        
        super().__init__(root=root, 
                         label_root=label_root, 
                         image_loader=image_loader, 
                         label_loader=label_loader, 
                         load_percentage=load_percentage,
                         keep_indices=keep_indices, 
                         filter_image_formats=filter_image_formats, 
                         filter_label_formats=filter_label_formats, 
                         dataset_name=dataset_name,
                         seed=seed)

        self.pseudolabels = np.array([-1 for _ in range(len(self))], dtype=np.int64)

    def subset(self, 
               indices: Optional[list[int]] = None,
               subset_name: Optional[str] = None,
               ids: Optional[list[int]] = None) -> BasicImageFolder:
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

        # select images
        if self.images is not None:
            subset_dataset.images = [subset_dataset.images[i] for i in indices]
        
        # select labels
        if self.labels is not None:
            subset_dataset.labels = [subset_dataset.labels[i] for i in indices]

        # select loaded images
        if self.loaded_images is not None:
            subset_dataset.loaded_images = [subset_dataset.loaded_images[i] 
                                            for i in indices]
        # select loaded labels
        if self.loaded_labels is not None:
            subset_dataset.loaded_labels = [subset_dataset.loaded_labels[i] 
                                            for i in indices]
        # select pseudolabels
        if self.pseudolabels is not None:
            subset_dataset.pseudolabels = np.array([subset_dataset.pseudolabels[i] 
                                            for i in indices], dtype=np.int64)

        # update ids and ids2indices (id:index) lookup table                                    
        subset_dataset.ids = [subset_dataset.ids[i] for i in indices]
        subset_dataset.ids2indices = \
        OrderedDict([(subset_dataset.ids[i], i) for i in range(len(indices))])
        

        subset_name = f"({subset_name})" if subset_name is not None else '(subset)'
        subset_dataset.name = self.dataset_name + subset_name

        return subset_dataset


    def __getitem__(self, index) -> SimpleNamespace:

        item = super().__getitem__(index)     

        return SimpleNamespace(image=item.image, 
                               label=item.label, 
                               pseudolabel=self.pseudolabels[index], 
                               id=self.ids[index], 
                               index=index)

    def update_pseudolabels(self,
                            values: Union[torch.LongTensor, List[int]],
                            indices: Optional[Union[torch.LongTensor, List[int]]] = None,
                            ids: Optional[Union[torch.LongTensor, List[int]]] = None):
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

        self.pseudolabels[indices] = values.cpu()
