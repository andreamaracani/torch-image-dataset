from typing import Optional, Tuple, List
import warnings
import os


def get_dir_names(path: str) -> List[str]:
    """ Returns the dir names inside directory specified by path. """

    return [dir_name for dir_name in os.listdir(path) if \
            os.path.isdir(os.path.join(path, dir_name))]


def get_file_names(path: str) -> List[str]:
    """ Returns the file names inside directory specified by path. """

    return [file_name for file_name in os.listdir(path) if \
            os.path.isfile(os.path.join(path, file_name))]


def match_file_names(path1: str,
                     path2: str,
                     formats_1: Optional[List[str]] = None,
                     formats_2: Optional[List[str]] = None, 
                     warn_mismatch: Optional[bool] = False) -> List[Tuple[str, str]]:
    """
        Find matching files not considering extensions.
        In particular a match when two files:
        
            path1/namefile.ext1, path2/namefile.ext2

        are found.
        
        Args:
            path1 (str): the first root path where to search files.
            path2 (str): the second root path where to search files.
            formats_1 (list[str], optional): if not None, consider just these formats in
            path1.
            formats_2 (list[str], optional): if not None, consider just these formats in
            path2.
            warn_mishmatch (bool, optional): True to warn when a file has no match.

        Returns:
            a list of tuples (file.ext1, file.ext2) of matching pairs.
    """
    names1  = sorted(get_file_names(path1))
    names2  = sorted(get_file_names(path2))
    pairs = []

    i1, i2 = 0, 0

    while i1 < len(names1) and i2 < len(names2):
        
        # separate filename from extension
        n1_noext, n1_ext = os.path.splitext(names1[i1])
        n2_noext, n2_ext = os.path.splitext(names2[i2])

        # remove . from extension
        n1_ext = n1_ext[1: ]
        n2_ext = n2_ext[1: ]

        # not allowed formats cases
        if formats_1 is not None and n1_ext not in formats_1:
            i1 += 1
        elif formats_2 is not None and n2_ext not in formats_2:
            i2 += 1

        # match case
        elif n1_noext == n2_noext:
            pairs.append((names1[i1], names2[i2]))
            i1 += 1
            i2 += 1

        # no match cases
        elif n1_noext < n2_noext:
            if warn_mismatch:
                msg = f"No matching found for file {names1[i1]} in folder " + \
                      path1
                warnings.warn(msg)
            i1 += 1
        else:
            if warn_mismatch:
                msg = f"No matching found for file {names2[i2]} in folder " + \
                      path2
                warnings.warn(msg)
            i2 += 1

    # some additional files have no match...
    while i1 < len(names1) and warn_mismatch:
        msg = f"No matching found for file {names1[i1]} in folder {path1}"
        warnings.warn(msg)
        i1 += 1

    while i2 < len(names2) and warn_mismatch:
        msg = f"No matching found for file {names2[i2]} in folder {path2}"
        warnings.warn(msg)
        i2 += 1

    return pairs


import numpy as np
from io import BytesIO


def array_to_bytes(x: np.ndarray) -> bytes:
    np_bytes = BytesIO()
    np.save(np_bytes, x, allow_pickle=True)
    return np_bytes.getvalue()


def bytes_to_array(b: bytes) -> np.ndarray:
    np_bytes = BytesIO(b)
    array = np.load(np_bytes, allow_pickle=True)

    if len(array.shape) == 0 and array.dtype == 'int':
        return int(array)
    
    return array