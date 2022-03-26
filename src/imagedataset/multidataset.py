from __future__ import annotations

from typing import List, Optional, Tuple, Union
import warnings

from imagedataset.imageloaders import LoaderPIL

from . import Interpolation
from . import AdvancedImageFolder, ImageLoader, from_database
import os
import copy
import pickle
from pathlib import Path

def multidataset_from_database(path, 
                               image_size: Optional[Tuple[int,int]] = None, 
                               load_labels: Optional[bool] = True) \
                               -> MultiDataset:
    """
        NOTE: 
            requires `lmdb` installed

        NOTE: 
            The structure of the databases/files should be the one made by
            MultiDataset's write_database function.

        Get a MultiDataset from some LMDB databases.

        Args:
            path (str): the path to the root of LMDB databases.
            image_size (Tuple, optional): A new image size for database images, None to
            keep original image size.
            load_labels (bool, optional): True to load labels to RAM, False to keep 
            labels inside the database.

    """

    with open(os.path.join(path, "__multidatasetobject__.p"), "rb") as file:
        multidataset = pickle.load(file)

    for domain in multidataset._domains:
        domain_path = os.path.join(path, domain)
        multidataset._datasets[domain] = from_database(domain_path, 
                                                       load_labels=load_labels, 
                                                       image_size=image_size)
    return multidataset


class MultiDataset:

    def __init__(self, 
                 root: str, 
                 image_loader: Optional[ImageLoader] = LoaderPIL(),
                 dataset_name: Optional[str] = None,
                 load_percentage: Optional[float] = None,
                 seed: Optional[int] = 123):
        """ 
            NOTE:
                IF load_percentage is not None, the dataset will be shuffled.

            A MultiDataset is a dict-like object with many datasets inside.
            To build a MultiDataset the folder structure should be:

            root/
              ├─────── dataset 1/
              │           ├── class 1/
              │           │      ├── image_1.ext
              │           │      ├── ...
              │           │      └── image_*.ext
              │           ├─── ...
              │           └── class N/
              │                  ├── image_1.ext
              │                  ├── ...
              │                  └── image_*.ext
              │
              ├─────── ...
              │
              └──────── dataset */
                          ├── class 1/
                          │      ├── image_1.ext
                          │      ├── ...
                          │      └── image_*.ext
                          ├─── ...
                          └── class N/
                                 ├── image_1.ext
                                 ├── ...
                                 └── image_*.ext

            Args:
                root (str): the root of the MultiDataset.
                image_loader (ImageLoader): the ImageLoader to load images from disk.
                dataset_name (str, optional): the name of the dataset.
                load_percentage (float, optional): the percentage of images to consider.
        """

        self._root            = root
        self._interpolation   = image_loader.interpolation
        self._image_size      = image_loader.size
        self._load_percentage = load_percentage
        self._dataset_name = dataset_name

        if not dataset_name:
            self._dataset_name = os.path.basename(os.path.dirname(os.path.join(root, "")))


        self._domains = [d for d in os.listdir(root)
                         if os.path.isdir(os.path.join(root, d))]
        self._paths   = [os.path.join(root, d) for d in self._domains]

        self._datasets = {}

        for domain in self._domains:
            self._datasets[domain]  = AdvancedImageFolder(
                                          root=os.path.join(root, domain),
                                          image_loader= copy.copy(image_loader),
                                          load_percentage=load_percentage,
                                          dataset_name=domain, 
                                          seed=seed) 
    
    def set_loader(self, loader: ImageLoader):
        """ Sets the loader to all datasets. """
        for dataset in self._datasets.values():
            dataset.set_loader(loader)
        return self
        
    def __repr__(self):
        """ Returns the repr of this object. """
        LINE_LENGTH = 90

        repr = ("=" * LINE_LENGTH + "\n")
        repr += f"MultiDataset[{self.dataset_name}]\n"
        repr += ("-" * LINE_LENGTH + "\n")
        for dataset in self._datasets.values():
            repr = repr + dataset.__repr__() + "\n"

        repr += ("=" * LINE_LENGTH + "\n")
        return repr

    def load_ram(self, 
                 numpy_images: Optional[bool] = False, 
                 num_workers: Optional[int] = 4) -> MultiDataset:
        """
            Loads all the datasets into RAM.
            Args:
                numpy_images (bool, optional): True to load images as numpy array, False
                to load them as jpg.
                num_workers (int, optional): the number of workers for the loading.
            Returns:
                self
        """
        for domain in self._domains:
            self._datasets[domain].load_ram(numpy_images=numpy_images, 
                                            num_workers=num_workers)
        return self

    def random_split(self,
                     train_percentage: float, 
                     test_percentage: Optional[float] = None, 
                     seed: Optional[int] = 123) -> Tuple[MultiDataset, MultiDataset]:
        """
            Splits all domain datasets into tain and test, accordingly to the 
            percentages

            Args:
                train_percentage (float): the percentage for training.
                test_percentage (float, optional): the percentage for test. If None it
                will be set to 1-train_percentage
                seed (int): seed for RNG.
            
            Returns:
                A pair of MultiDataset, one for train, one for test.
        """
        if test_percentage is None:
            test_percentage = 1 - train_percentage

        percentages = [train_percentage, test_percentage]

        train_dataset = copy.copy(self)
        train_dataset._datasets = {}
        test_dataset  = copy.copy(self)
        test_dataset._datasets = {}

        train_dataset._dataset_name += " (train)"
        test_dataset._dataset_name  += " (test)"

        for domain in self._domains:
            dataset = self._datasets[domain] 
            train, test = dataset.random_split(percentages=percentages,
                                               split_names = ["train", "test"],
                                               seed = seed)
            train_dataset._datasets[domain] = train
            test_dataset._datasets[domain]  = test

        return train_dataset, test_dataset

    @property
    def image_size(self) -> Tuple[int, int]:
        """ Get the current image size. """
        return self._image_size
 
    @image_size.setter
    def image_size(self, image_size: Tuple[int, int]):
        """ Set the image_size to all datasets. """
        for dataset in self._datasets.values():
            dataset.image_size = image_size
        self._image_size = image_size

    def items(self):
        """ Get the items. """
        return self._datasets.items()

    def keys(self):
        """ Get the keys. """
        return self._datasets.keys()

    def values(self):
        """ Get the values. """
        return self._datasets.values()

    def __getitem__(self, domain: Union[str, int]):
        """ Get a dataset from the domain name. """

        if isinstance(domain, str):
            return self._datasets[domain]
        
        return self._datasets[self._domains[domain]]

    def __len__(self):
        """ Get the number of datasets. """
        return len(self._datasets)

    def __iter__(self):
        """ Get the iterator. """
        return self._datasets.__iter__()

    def n_classes(self) -> int:
        """ Get the number of classes of the dataset. """
        first_key = list(self.keys())[0]
        return self._datasets[first_key].n_classes()

    @property
    def domains(self) -> List[str]:
        """ Get the domains. """
        return self._domains
    
    @property
    def interpolation(self) -> Interpolation:
        """ Get the interpolation. """
        return self._interpolation

    @property
    def dataset_name(self) -> str:
        """ Get the dataset name. """
        return self._dataset_name

    def write_database(self, 
                       path: str, 
                       numpy_images: Optional[bool] = False, 
                       num_workers: Optional[int] = 4, 
                       map_size: Optional[int] = int(1e13),
                       verbose: Optional[bool] = False):
        """ 
            Write the current MultiDatset to some LMDB databases (one for each domain).

            Args:
                path (str): the root path for databases.
                numpy_images (bool, optional): True to save numpy_images in the database,
                False to save jpg.
                num_workers (int, optional): The number of workers to save the database.
                map_size (int, optional): The maximum kB size of a single database.
                verbose (bool, optional): True to print status.
        """

        if os.path.exists(path):
            msg = "Database at {path} aready exists! Skipping the writing."
            warnings.warn(msg)
            return
            
        Path(path).mkdir(parents=True, exist_ok=True)

        # We do not want to save self._datasets inside the object so we remove it, 
        # we save the object and we put it back.

        tmp_datasets = self._datasets
        self._datasets = {}

        with open(os.path.join(path, "__multidatasetobject__.p"), "wb") as file:
            pickle.dump(self, file)

        self._datasets = tmp_datasets

        # Save all domain_datasets as databases.
        for domain, dataset in self._datasets.items():
            if verbose:
                print(f"Starting writing dataset: {domain}")

            dataset.write_database(path=os.path.join(path, domain),
                                   numpy_images=numpy_images,
                                   num_workers=num_workers,
                                   map_size=map_size,
                                   verbose=verbose)