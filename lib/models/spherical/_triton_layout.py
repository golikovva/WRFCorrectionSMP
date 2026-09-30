"""Layout constraints for strides specialized by inline Triton operations."""

import torch
from torch import Tensor


def fix_triton_strides(value: Tensor) -> Tensor:
    """Keep traced strides consistent with an inline Triton launch.

    Inductor may pad an intermediate after its strides have been captured as
    kernel constants. An identity as_strided makes the producer layout
    observable. Inductor does not enforce exact strides for a view producer,
    so views and tensors with an offset first receive an independent
    contiguous producer. Ordinary
    contiguous tensors only receive a view and incur no tensor copy.

    This helper runs inside triton_op, where AOT can read storage_offset
    metadata. Offset tensors are materialized before as_strided, so its
    omitted offset is zero even when functionalization erased the view base.
    Apply unconditionally: AOT decompositions also need this constraint.
    """
    if value._base is not None or not value.is_contiguous() or value.storage_offset() != 0:
        value = value.clone(memory_format=torch.contiguous_format)
    return value.as_strided(value.shape, value.stride())
