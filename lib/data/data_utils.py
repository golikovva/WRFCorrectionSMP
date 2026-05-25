import functools
import warnings
import torch
import numpy as np
from typing import Callable, List, Dict, Tuple, Any

from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import PackedSequence


def none_consistent_collate(batch):
    elem = batch[0]
    if isinstance(elem, type(None)):
        return None
    elif isinstance(elem, np.datetime64):
        return np.array(batch)
    elif isinstance(elem, tuple):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.
        return [none_consistent_collate(samples) for samples in transposed]  # Backwards compatibility.
    else:
        return default_collate(batch)


def numpy_collate(batch):
    transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.
    return np.array([np.array(samples) for samples in transposed])  # Backwards compatibility.


def none_consistent_numpy_collate(batch):
    elem = batch[0]
    if isinstance(elem, type(None)):
        return None
    elif isinstance(elem, np.datetime64):
        return np.array(batch)
    elif isinstance(elem, tuple):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.
        return [none_consistent_numpy_collate(samples) for samples in transposed]  # Backwards compatibility.
    else:
        return numpy_collate(batch)


class TestSampler:
    def __init__(self, data_len, seq_len):
        self.data_len = data_len
        self.seq_len = seq_len

    def __len__(self):
        return self.data_len

    def __iter__(self):
        return iter(range(0, len(self), self.seq_len))


def find_files(directory, pattern):
    import os, fnmatch
    flist = []
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                filename = filename.replace('\\', '/')
                flist.append(filename)
    return flist


class Sampler:
    def __init__(self, days, shuffle=False):
        self.days = days
        self.shuffle = shuffle

    def __len__(self):
        return len(self.days)

    def __iter__(self):
        ids = np.arange(len(self.days))
        if self.shuffle:
            np.random.shuffle(ids)
        for i in ids:
            yield self.days[i]

def _contains_none(x):
    """Recursively check if a sample contains None anywhere."""
    if x is None:
        return True
    if isinstance(x, (list, tuple, set)):
        return any(_contains_none(v) for v in x)
    if isinstance(x, dict):
        return any(_contains_none(v) for v in x.values())
    # We assume ndarrays/tensors here never contain Python None as payload.
    return False

def ignore_warnings(category: Warning):
    def ignore_warnings_decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=category)
                return func(*args, **kwargs)
        return wrapper
    return ignore_warnings_decorator

@ignore_warnings(category=UserWarning)
def variable_len_collate(batch):
    """
    Custom collate function for handling batches with None values and variable-length sequences.

    Args:
        batch (list): A list of data samples.
    elif isinstance(elem, torch.Tensor) and any(len(b) != len(elem) for b in batch):
    Returns:
        Collated batch data, which can be None, a list of datetime64 objects, a packed sequence of tensors,
        or a default collated batch depending on the input batch type.
    """
    is_list = False
    if isinstance(batch[0], (list, tuple, set)):
        is_list = True
        total_elems = len(batch[0])

    batch = [b for b in batch if not _contains_none(b)]
    if len(batch) == 0:
        if is_list:
            return [None]*total_elems
        return None  # training loop should skip this batch

    elem = batch[0]
    if isinstance(elem, np.datetime64):
        return np.array(batch)
    
    if isinstance(elem, np.ndarray):
        return variable_len_collate([torch.from_numpy(item) for item in batch])
    
    if isinstance(elem, torch.Tensor):
        try:
            return torch.nested.to_padded_tensor(torch.nested.nested_tensor(batch), padding=0.0)
        except RuntimeError:
            # not ragged in the leading dim; let default handle strict stacking
            return default_collate(batch)
        # return torch.nested.to_padded_tensor(torch.nested.nested_tensor(batch), padding=0.0)
        # return pack_sequence(batch, enforce_sorted=False)

    if isinstance(elem, (list, tuple)):
        # check to make sure that the elements in batch have consistent size
        # it = iter(batch)
        # elem_size = len(next(it))
        # if not all(len(elem) == elem_size for elem in it):
        #     raise RuntimeError('each element in list of batch should be of equal size')
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.
        return [variable_len_collate(samples) for samples in transposed]  # Backwards compatibility.
    
    if isinstance(elem, dict):
        keys = elem.keys()
        return {k: variable_len_collate([d[k] for d in batch]) for k in keys}
    return default_collate(batch)


def transform_packed_sequence_multiple(
    packed: PackedSequence,
    transforms: List[Tuple[Callable[[torch.Tensor, Any], torch.Tensor], Tuple[Any, ...], Dict[str, Any]]]
) -> PackedSequence:
    """
    Последовательно применяет несколько функций преобразования к данным внутри PackedSequence.
    
    Args:
        packed (PackedSequence): Исходный PackedSequence.
        transforms (List[Tuple[Callable, Tuple, Dict]]): Список преобразований, где:
            - первый элемент: функция преобразования,
            - второй элемент: кортеж позиционных аргументов для функции,
            - третий элемент: словарь именованных аргументов для функции.
    
    Returns:
        PackedSequence: Новый PackedSequence с преобразованными данными.
    """
    data = packed.data  # Исходные данные PackedSequence

    # Применяем каждую функцию последовательно
    for transform_fn, args, kwargs in transforms:
        data = transform_fn(data, *args, **kwargs)

    # Создаем новый PackedSequence с преобразованными данными
    return PackedSequence(data, packed.batch_sizes, packed.sorted_indices, packed.unsorted_indices)


def get_novaya_zemlya_mask(fill_value=torch.nan, return_vertices=False):
    nz_vertices = np.array([
        [120, 120],
        [70, 130],
        [10, 190],
        [0, 240],
        [40, 280],
        [160, 160],
    ])
    nz_polygon_array = torch.from_numpy(create_polygon([210, 280], nz_vertices))
    nz_polygon_array = torch.where(nz_polygon_array == 0, fill_value, 1)
    if not return_vertices:
        return nz_polygon_array
    return nz_polygon_array, nz_vertices


def check(p1, p2, base_array):
    """
    Uses the line defined by p1 and p2 to check array of
    input indices against interpolated value

    Returns boolean array, with True inside and False outside of shape
    """
    idxs = np.indices(base_array.shape)  # Create 3D array of indices

    p1 = p1.astype(float)
    p2 = p2.astype(float)

    # Calculate max column idx for each row idx based on interpolated line between two points
    max_col_idx = (idxs[0] - p1[0]) / (p2[0] - p1[0]) * (p2[1] - p1[1]) + p1[1]
    sign = np.sign(p2[0] - p1[0])
    return idxs[1] * sign <= max_col_idx * sign


def create_polygon(shape, vertices):
    """
    Creates np.array with dimensions defined by shape
    Fills polygon defined by vertices with ones, all other values zero"""
    base_array = np.zeros(shape, dtype=float)  # Initialize your array of zeros

    fill = np.ones(base_array.shape) * True  # Initialize boolean array defining shape fill

    # Create check array for each edge segment, combine into fill array
    for k in range(vertices.shape[0]):
        fill = np.all([fill, check(vertices[k - 1], vertices[k], base_array)], axis=0)

    # Set all values inside polygon to one
    base_array[fill] = 1

    return base_array
