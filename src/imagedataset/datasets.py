from __future__ import annotations

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

from .utils import (
    get_dir_names, 
    get_file_names, 
    match_file_names, 
    array_to_bytes, 
    bytes_to_array
)

from .dataloaders import CudaLoader, get_collate
from .operations import  ImageLoader, FileLoader
from .imageloaders import LoaderPIL


def from_database(path, load_labels=True) -> Union[BasicImageFolder, AdvancedImageFolder]:
    """
        NOTE: requires `lmdb` installed
        Get a BasicImageFolder or an AdvanceImageFolder from a LMDB database file.

        Args:
            path (str): the path to the LMDB.
            load_labels (bool, optional): True to load labels to RAM, False to keep labels
            inside the database.
    """

    import lmdb

    env = lmdb.open(path, readonly=True, lock=True)
    txn = env.begin(write=False)

    dataset = pickle.loads(txn.get('__object__'.encode()))

    if load_labels:
        dataset.loaded_labels = []
        for i in range(len(dataset)):
            label = bytes_to_array(txn.get(f"label{i}".encode()))
            dataset.loaded_labels.append(label)  

    dataset._initLMDB_env(path)

    return dataset


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

        # True if we have external labels, False if the labels are the names of folders.
        self.external_labels = label_root is not None
        self.from_database = False

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
        

        # SETUP DATASET
        
        self.classes       = None  # class names list
        self.class2index   = None  # dict {class_name:index}
        self.images        = None  # list of the paths to images
        self.labels        = None  # list of the paths to labels
        self.loaded_images = None  # list of images (loaded)
        self.loaded_labels = None  # list of labels (loaded)
        self.images_loaded_as_numpy = False # True if images have been loaded as numpy.
                                            # False if images have been loaded as bytes.
        self.images_database_as_numpy = False

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
        return f'<{self.__class__.__name__} len={self.__len__()} ' \
             + f'name={self.dataset_name}>'


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

        # MODE: NOT DATABASE
        if not self.from_database:

            # IMAGES 
            if self.loaded_images is not None:                  # 1) NUMPY IMAGES (RAM)
                if self.images_loaded_as_numpy:
                    image = self.loaded_images[index]
                else:                                           # 2) BYTES IMAGES (RAM)
                    image = self.image_decoder(self.loaded_images[index])
            else:                                               # 3) IMAGES (DISK)
                image = self.image_loader(self.images[index])

            # LABELS
            if self.loaded_labels is not None:                  # 1) LABELS (RAM)
                label = self.loaded_labels[index]
            else:                                               # 2) LABELS (DISK)
                label = self.label_loader(self.labels[index])


        # MODE: DATABASE
        else:

            if self.txn is None:  
                self._initLMDB_txn()

            # IMAGE
            if self.loaded_images is not None:
                if self.images_loaded_as_numpy:              # 1) NUMPY IMAGES (RAM)
                    image = self.loaded_images[index]
                else:                                        # 2) BYTES IMAGES (RAM)
                    image = self.image_decoder(self.loaded_images[index])
            else:
                if self.images_database_as_numpy:            # 3) NUMPY IMAGES (DATABASE)
                    image = bytes_to_array(self.txn.get(f"image{index}".encode()))
                else:                                        # 4) BYTES IMAGES (DATABASE)
                    image = self.image_decoder(self.txn.get(f"image{index}".encode()))

            # LABEL
            if self.loaded_labels is not None:               # 1) LABELS (RAM)
                label = self.loaded_labels[index]
            else:                                            # 2) LABELS (DATABASE)
                label = bytes_to_array(self.txn.get(f"label{index}".encode()))

        return SimpleNamespace(image=image, label=label)


    def load_ram(self,
                 numpy_images: Optional[bool] = False,
                 num_workers: Optional[int] = 4):
        """ 
            Loads to RAM the entire dataset.

            Args:
                numpy_images (bool, optioanl): True to load images as np.ndarray in 
                memory (it will speed up loading but it consumes more memory).
                num_workers (int, optional): The number of workers for the loading.
        """

        # keep track if images have been loaded as numpy
        self.images_loaded_as_numpy = numpy_images

        # new lables/images loaded in memory (in lists).
        loaded_images = []
        loaded_labels = []

        # get a collate function that does not alter data and do not convert to tensor.
        collate_fn = get_collate(to_tensor=False, 
                                 images_last2first=False)

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
        self.loaded_images = loaded_images

        # set labels
        self.loaded_labels = loaded_labels


    def cudaloader(self,
                   mean: Optional[tuple] = None,
                   std: Optional[tuple] = None,
                   fp16: Optional[bool] = False,
                   channels_last_shape: Optional[bool] = False,
                   channels_last_memory: Optional[bool] = False,
                   batch_size: Optional[int] = 1,
                   shuffle: bool = False, 
                   sampler: Optional[Sampler[int]] = None,
                   batch_sampler: Optional[Sampler[Sequence[int]]] = None,
                   num_workers: int = 0,
                   pin_memory: bool = False, 
                   drop_last: bool = False) -> CudaLoader:

        """ 
            NOTE:
                images contain FLOATS32 or 16 without normalization, so their value will
                be between 0 and 255.
                With mean and std it is possible to normalize them.

            NOTE:
                other tensors contain just uint8 or int64 integers.
            
            Get a torch.utils.data.DataLoader object associated with this dataset 
            wrapped inside a CudaLoader with prefatching and fast load to gpu.

            Args:
                mean (tuple, optional): the mean to subtract to images.
                std (tuple, optional): the std to divide the images.
                fp15 (bool, optional): True to convert to half precision.
                channels_last_shape (bool, optioanl): True to return images with the 
                channel-last shape, False oterwise.
                channels_last_memory (bool, optional): True to return tensors with the 
                memory in channel last format, False to return tensors with contiguos
                memory format.
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

        if channels_last_memory:
            memory_format = torch.channels_last
        else:
            memory_format = torch.contiguous_format

        collate_fn = get_collate(memory_format, not channels_last_shape, to_tensor=True)
        dataloader = DataLoader(self, 
                                batch_size=batch_size, 
                                shuffle=shuffle, 
                                sampler=sampler, 
                                batch_sampler=batch_sampler, 
                                num_workers=num_workers, 
                                pin_memory=pin_memory, 
                                drop_last=drop_last,
                                collate_fn=collate_fn)
        return CudaLoader(loader=dataloader, mean=mean, std=std, fp16=fp16)



    def dataloader(self,
                   channels_last_shape: Optional[bool] = False,
                   channels_last_memory: Optional[bool] = False,
                   batch_size: Optional[int] = 1,
                   shuffle: bool = False, 
                   sampler: Optional[Sampler[int]] = None,
                   batch_sampler: Optional[Sampler[Sequence[int]]] = None,
                   num_workers: int = 0,
                   pin_memory: bool = False, 
                   drop_last: bool = False) -> Union[DataLoader, CudaLoader]:
        """ 
            NOTE:
                the output tensors contain just uint8 or int64 integers.

            Get a torch.utils.data.DataLoader object associated with this dataset.

            Args:
                channels_last_shape (bool, optioanl): True to return images with the 
                channel-last shape, False oterwise.
                channels_last_memory (bool, optional): True to return tensors with the 
                memory in channel last format, False to return tensors with contiguos
                memory format.
                batch_size (int, optional): the batch size.
                shuffle (bool, optional): True to shuffle indices.
                sampler (Sampler, optional): a sampler for indices.
                batch_sampler (Sampler, optional): a sampler for batches.
                num_workers (int, optional): the number of concurrent workers.
                pin_memory (bool, optional): True to pin gpu memory.
                drop_last (bool, optional): True to drop last batch if incomplete.

            Returns: 
                a torch.utils.data.DataLoader object.
        """
                
        if channels_last_memory:
            memory_format = torch.channels_last
        else:
            memory_format = torch.contiguous_format
        collate_fn = get_collate(memory_format, not channels_last_shape, to_tensor=True)

        dataloader = DataLoader(self, 
                                batch_size=batch_size, 
                                shuffle=shuffle, 
                                sampler=sampler, 
                                batch_sampler=batch_sampler, 
                                num_workers=num_workers, 
                                pin_memory=pin_memory, 
                                drop_last=drop_last,
                                collate_fn=collate_fn)

        return dataloader

    
    def write_database(self,
                       path: str,
                       numpy_images: Optional[bool] = False,
                       num_workers: Optional[int] = 1,
                       map_size: Optional[int] = int(1e12)):
        """ 
            NOTE: requires `lmdb` installed.

            Writes the entire dataset inside a LMDB database.

            Args:
                path (str): the path where to save the database.
                numpy_images (bool, optional): True to save images as numpy arrays in the
                database.
                num_workers (int, optional): the number of workers for the saving.
                map_size (int, optional): the max number of kB allowed for the database.

        """
        import lmdb
        self.images_database_as_numpy = numpy_images

        env = lmdb.open(path, map_size=map_size, lock=False)
        txn = env.begin(write=True)
        collate_fn = get_collate(to_tensor=False, 
                                 images_last2first=False)

        dataloader = DataLoader(self, 
                                batch_size=1, 
                                num_workers=num_workers, 
                                shuffle=False, 
                                drop_last=False,
                                collate_fn=collate_fn)
            
        for idx, batch in enumerate(dataloader):
            image = batch.image[0]
            if self.images_database_as_numpy:
                image = array_to_bytes(image)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image = cv2.imencode(".jpg", image)[1].tobytes()

            label = array_to_bytes(batch.label[0])
            txn.put(('image'+ str(idx)).encode(), image)
            txn.put(('label'+ str(idx)).encode(), label)
     
        tmp_loaded_images = self.loaded_images
        tmp_loaded_labels = self.loaded_labels 
        self.loaded_images = None
        self.loaded_labels = None

        encoded_object = pickle.dumps(self)
        txn.put('__object__'.encode(), encoded_object)

        self.loaded_images = tmp_loaded_images
        self.loaded_labels = tmp_loaded_labels

        txn.commit()
        env.close()


    def _initLMDB_env(self, path):
        """ Initializes the database from a path. """
        import lmdb
        self.from_database = True
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

        self.pseudolabels[indices] = values
