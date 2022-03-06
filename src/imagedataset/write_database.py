from lmdb import Environment
import lmdb
from typing import Callable, Tuple, TypeVar, Optional
import cv2
import os
import warnings
import pickle 
from multiprocessing import Process

T = TypeVar("T")

def _write_samples(lmdb_env: Environment,
                   samples: list[str],
                   targets: list[T],
                   target_loader: Optional[Callable[[T], bytes]] = None, 
                   from_idx: Optional[int] = None, 
                   to_idx: Optional[int] = None,
                   image_format: Optional[str] = "jpg", 
                   convert_other_img_formats: Optional[bool] = True, 
                   image_size: Optional[Tuple[int, int]] = None, 
                   interpolation: Optional[int] = cv2.INTER_CUBIC,
                   warn_ignored: Optional[bool] = False):
        """
            Writes samples and targets into a LMDB database. The key will be:

                'X{idx}' for samples  
                'Y{idx}' for targets
            
            Just samples/targets in the interval [from_idx, to_idx) will be inserted.

            Args:
                lmdb_env (Environment): the environment of lmdb.
                samples (list[str]): the samples of a BasicImageFolder.
                targets (list[T]): the targets of a BasicImageFolder.
                target_loader (Callable): a function that take a target and converts it
                to bytes for the serialization.
                from_idx (int, optional): the starting idx (included).
                to_idx (int, optional): the ending idx (excluded).
                image_format (str, optional): the format of images.
                convert_other_img_formats (bool, optional): if True all img formats will
                be converted to the `image_format`, otherwise they will be ignored.
                image_size (tuple, optional): if specified, all images will be resized to
                this size.
                interpolation (int, optional): the opencv interpolation integer for 
                resizing.
                warn_ignored (bool, optional): True to warn when images are skipped.
                
        """

        txn = lmdb_env.begin(write=True)

        for idx in range(from_idx, to_idx):
            image_path = samples[idx]
            target = targets[idx]

            image_extension = os.path.splitext(image_path)[1][1:]

            # skipping image
            if image_extension != image_format and not convert_other_img_formats:

                if warn_ignored:
                    msg = f"Skipping image at path {image_path}, no matching format!"
                    warnings.warn(msg)

                continue
            
            # serialize target
            if target_loader is None: 
                target = str(target).encode()
            else: 
                target = target_loader(target)


            # serialize image
            image = cv2.imread(image_path)

            if image_size is not None:
                image = cv2.resize(image, dsize=image_size, interpolation=interpolation)

            image = cv2.imencode("." + image_format, image)[1].tobytes()

            # insert image and target
            txn.put(('X'+ str(idx)).encode(), image)
            txn.put(('Y'+ str(idx)).encode(), target)
        
        txn.commit()


def _write_info(dataset, img_format: str, lmdb_env: Environment):
    """
        Writes to the database info about the dataset.

        Args:
            dataset (BasicImageFolder): the dataset.
            img_format (str): the format of images.
            lmdb_env (Environment): the enviroment of LMDB.
    """
    txn = lmdb_env.begin(write=True)
    
    # get info and serialize them
    length = str(len(dataset)).encode()
    classes = pickle.dumps(dataset.classes)
    classes2idx = pickle.dumps(dataset.class2idx)
    img_format = img_format.encode()

    txn.put("__length__".encode(), length)
    txn.put("__classes__".encode(), classes)
    txn.put("__classes2idx__".encode(), classes2idx)
    txn.put("__format__".encode(), img_format)

    txn.commit()


def read_info(lmdb_path: str):
    """
        Read to the database info about the dataset.

        Args:
            lmdb_env (Environment): the enviroment of LMDB.
    """
    txn = lmdb.open(lmdb_path, readonly=True).begin(write=False)

    length = txn.get("__length__".encode())
    classes = txn.get("__classes__".encode())
    classes2idx = txn.get("__classes2idx__".encode())
    img_format = txn.get("__format__".encode())

    length = int(length)
    classes = pickle.loads(classes)
    classes2idx = pickle.loads(classes2idx)
    img_format = img_format.decode("utf-8")

    return length, classes, classes2idx, img_format


def _write_single_process(dataset, 
                          path: str,
                          map_size: Optional[int] = 100000000,
                          target_loader: Optional[Callable[[T], bytes]] = None, 
                          image_format: Optional[str] = "jpg", 
                          convert_other_img_formats: Optional[bool] = True, 
                          image_size: Optional[Tuple[int, int]] = False, 
                          interpolation: Optional[int] = cv2.INTER_CUBIC,
                          warn_ignored: Optional[bool] = False):

    lmdb_env = lmdb.open(path, map_size=map_size, lock=True)

    samples = dataset.samples
    targets = dataset.targets

    _write_samples(lmdb_env=lmdb_env,
                   samples=samples,
                   targets=targets,
                   target_loader=target_loader, 
                   from_idx=0, 
                   to_idx=len(dataset),
                   image_format=image_format,
                   convert_other_img_formats=convert_other_img_formats,
                   image_size=image_size,
                   interpolation=interpolation,
                   warn_ignored=warn_ignored)

    _write_info(dataset=dataset, img_format=image_format, lmdb_env=lmdb_env)


def _write_multi_process(dataset, 
                         path: str,
                         num_processes: Optional[int] = 1,
                         map_size: Optional[int] = 100000000,
                         target_loader: Optional[Callable[[T], bytes]] = None, 
                         image_format: Optional[str] = "jpg", 
                         convert_other_img_formats: Optional[bool] = True, 
                         image_size: Optional[Tuple[int, int]] = None, 
                         interpolation: Optional[int] = cv2.INTER_CUBIC,
                         warn_ignored: Optional[bool] = False):

    total_samples = len(dataset)
    samples_per_process = total_samples // num_processes

    # get idx start and end
    idx_start = [i  for i in range(0, total_samples, samples_per_process)]
    if len(idx_start) > num_processes:
            idx_start = idx_start[:-1]
        
    idx_end = [idx_start[i+1] for i in range(len(idx_start)-1)] + [total_samples]

    samples = dataset.samples
    targets = dataset.targets

    lmdb_env = lmdb.open(path, map_size=map_size, lock=True)
    processes = []

    for i in range(num_processes):

        args = (lmdb_env, samples, targets, target_loader, idx_start[i], idx_end[i], 
                image_format, convert_other_img_formats, image_size, interpolation, 
                warn_ignored)

        processes.append(Process(target=_write_samples, args=args))


    for p in processes:
        p.start()

    for p in processes:
        p.join()

    _write_info(dataset=dataset, img_format=image_format, lmdb_env=lmdb_env)


def write(dataset, 
          path: str,
          num_processes: Optional[int] = 1,
          map_size: Optional[int] = 100000000,
          target_loader: Optional[Callable[[T], bytes]] = None, 
          image_format: Optional[str] = "jpg", 
          convert_other_img_formats: Optional[bool] = True, 
          image_size: Optional[Tuple[int, int]] = None, 
          interpolation: Optional[int] = cv2.INTER_CUBIC,
          warn_ignored: Optional[bool] = False):
    """
        Writes the dataset to a LMDB database located in `path`.

        Args:
            dataset (BasicImageFolder): the dataset to write.
            path (str): the target path.
            num_processes (int, optional): the number of processes.
    """
    args = locals()

    if num_processes > 1:
        return _write_multi_process(**args)
    else:
        del args["num_processes"]
        return _write_single_process(**args)

