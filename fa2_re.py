import ctypes
import math
import os

import torch
import triton
import triton.language as tl


class _DLDeviceRecord(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class _DLScalarType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _DLTensorRecord(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDeviceRecord),
        ("ndim", ctypes.c_int),
        ("dtype", _DLScalarType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedRecord(ctypes.Structure):
    pass


_TensorFinalizer = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedRecord))
_DLManagedRecord._fields_ = [
    ("dl_tensor", _DLTensorRecord),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", _TensorFinalizer),
]
_managed_owners = {}
_managed_cache = {}


def _cuda_status(rc, op):
    if rc:
        raise RuntimeError(f"{op}: CUDA error {rc}")


@_TensorFinalizer
def _finalize_managed_record(record_pointer):
    """Release or cache storage while its DLPack metadata remains alive."""
    allocation_owner = _managed_owners.pop(
        ctypes.addressof(record_pointer.contents), None
    )
    if allocation_owner is None:
        return
    (
        cuda_driver,
        allocation_address,
        cuda_context,
        managed_record,
        tensor_shape,
        cache_key,
    ) = allocation_owner
    if cache_key is not None:
        _managed_cache.setdefault(cache_key, []).append(allocation_address)
        return
    if cuda_driver.cuCtxPushCurrent_v2(cuda_context) == 0:
        cuda_driver.cuCtxSynchronize()
        cuda_driver.cuMemFree_v2(allocation_address)
        previous_context = ctypes.c_void_p()
        cuda_driver.cuCtxPopCurrent_v2(ctypes.byref(previous_context))


_is_dlpack_capsule = ctypes.pythonapi.PyCapsule_IsValid
_is_dlpack_capsule.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_is_dlpack_capsule.restype = ctypes.c_int
_dlpack_payload = ctypes.pythonapi.PyCapsule_GetPointer
_dlpack_payload.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_dlpack_payload.restype = ctypes.c_void_p
_CapsuleFinalizer = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


@_CapsuleFinalizer
def _finalize_dlpack_capsule(tensor_capsule):
    """Dispose of a capsule only when its tensor has not been consumed."""
    if _is_dlpack_capsule(tensor_capsule, b"dltensor"):
        _finalize_managed_record(
            ctypes.cast(
                _dlpack_payload(tensor_capsule, b"dltensor"),
                ctypes.POINTER(_DLManagedRecord),
            )
        )


def _allocate_managed_storage(
    tensor_shape, storage_dtype, cuda_device, prefer_host=False, retain_allocation=False
):
    """Expose managed CUDA storage as an owning, contiguous Torch tensor."""
    storage_bits = {torch.float16: 16, torch.float32: 32}[storage_dtype]
    cuda_driver = ctypes.CDLL("libcuda.so.1")
    cuda_driver.cuMemAllocManaged.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.c_uint,
    ]
    cuda_driver.cuMemFree_v2.argtypes = [ctypes.c_uint64]
    cuda_driver.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    cuda_driver.cuCtxPushCurrent_v2.argtypes = [ctypes.c_void_p]
    cuda_driver.cuCtxPopCurrent_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    cuda_context = ctypes.c_void_p()
    _cuda_status(
        cuda_driver.cuCtxGetCurrent(ctypes.byref(cuda_context)), "cuCtxGetCurrent"
    )
    allocation_bytes = math.prod(tensor_shape) * (storage_bits // 8)
    cache_key = (
        (allocation_bytes, cuda_device.index, bool(prefer_host))
        if retain_allocation
        else None
    )
    cached_blocks = _managed_cache.get(cache_key) if retain_allocation else None
    if cached_blocks:
        allocation_address = cached_blocks.pop()
    else:
        allocation_address = ctypes.c_uint64()
        _cuda_status(
            cuda_driver.cuMemAllocManaged(
                ctypes.byref(allocation_address), allocation_bytes, 1
            ),
            "cuMemAllocManaged",
        )
        if prefer_host:
            cuda_driver.cuMemAdvise.argtypes = [
                ctypes.c_uint64,
                ctypes.c_size_t,
                ctypes.c_int,
                ctypes.c_int,
            ]
            _cuda_status(
                cuda_driver.cuMemAdvise(allocation_address, allocation_bytes, 3, -1),
                "cuMemAdvise",
            )
            _cuda_status(
                cuda_driver.cuMemAdvise(
                    allocation_address, allocation_bytes, 5, cuda_device.index
                ),
                "cuMemAdvise",
            )
    shape_storage = (ctypes.c_int64 * len(tensor_shape))(*tensor_shape)
    managed_record = _DLManagedRecord(
        _DLTensorRecord(
            ctypes.c_void_p(allocation_address.value),
            _DLDeviceRecord(2, cuda_device.index),
            len(tensor_shape),
            _DLScalarType(2, storage_bits, 1),
            shape_storage,
            None,
            0,
        ),
        None,
        _finalize_managed_record,
    )
    # The capsule contains raw addresses; retain the metadata until its deleter runs.
    _managed_owners[ctypes.addressof(managed_record)] = (
        cuda_driver,
        allocation_address,
        cuda_context,
        managed_record,
        shape_storage,
        cache_key,
    )
    create_capsule = ctypes.pythonapi.PyCapsule_New
    create_capsule.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    create_capsule.restype = ctypes.py_object
    tensor_capsule = create_capsule(
        ctypes.addressof(managed_record),
        b"dltensor",
        ctypes.cast(_finalize_dlpack_capsule, ctypes.c_void_p),
    )
    return torch.utils.dlpack.from_dlpack(tensor_capsule)


def release_pool():
    """Synchronously release allocations retained by HOST_POOL."""
    if not _managed_cache:
        return
    cuda_driver = ctypes.CDLL("libcuda.so.1")
    cuda_driver.cuMemFree_v2.argtypes = [ctypes.c_uint64]
    torch.cuda.synchronize()
    for cached_addresses in _managed_cache.values():
        for allocation_address in cached_addresses:
            cuda_driver.cuMemFree_v2(allocation_address)
    _managed_cache.clear()

def _split_dim(padded_dim, max_part=128):
    """Describe a padded feature axis with at most three MMA-sized pieces."""
    pieces = []
    remaining = padded_dim
    for width in (128, 64, 32, 16):
        if not (width <= max_part and remaining >= width):
            continue
        count, remaining = divmod(remaining, width)
        pieces.extend([width] * count)
    missing = 3 - len(pieces)
    return tuple(pieces + [0] * missing)

@triton.jit
def _operand_product(lhs, rhs, mode: tl.constexpr, acc=None):
    """Convert the operands before issuing the requested dot-product variant."""
    if mode == "fp16":
        lhs16 = lhs.to(tl.float16)
        rhs16 = rhs.to(tl.float16)
        return tl.dot(lhs16, rhs16, acc)
    else:
        lhs32 = lhs.to(tl.float32)
        rhs32 = rhs.to(tl.float32)
        return tl.dot(lhs32, rhs32, acc, input_precision=mode)

@triton.jit
def _residual_product(lhs, rhs, total):
    """Compute the low component first, then fold in the leading half."""
    # Preparing rhs is independent of the hi/lo decomposition of lhs.
    rhs16 = rhs.to(tl.float16)
    hi = lhs.to(tl.float16)
    lo = (lhs - hi.to(tl.float32)).to(tl.float16)
    product = tl.dot(lo, rhs16)
    product = tl.dot(hi, rhs16, product)
    return total + product

@triton.jit
def _read_feature_major(
    src, row_ids, valid,
    col0: tl.constexpr, width: tl.constexpr,
    pitch: tl.constexpr, masked: tl.constexpr,
):
    """Present contiguous feature spans as the left axis of a loaded tile."""
    cols = col0 + tl.arange(0, width)
    row_offsets = row_ids[None, :] * pitch
    ptrs = src + row_offsets + cols[:, None]
    if masked:
        return tl.load(ptrs, valid[None, :], other=0)
    else:
        return tl.load(ptrs)

@triton.jit
def _read_token_major(
    src, row_ids, valid,
    col0: tl.constexpr, width: tl.constexpr,
    pitch: tl.constexpr, masked: tl.constexpr,
):
    """Load the same storage with token indices on the left axis."""
    cols = col0 + tl.arange(0, width)
    row_offsets = row_ids[:, None] * pitch
    ptrs = src + row_offsets + cols[None, :]
    if masked:
        return tl.load(ptrs, valid[:, None], other=0)
    else:
        return tl.load(ptrs)

@triton.jit
def _write_feature_slice(
    dst, row_ids, row_ok, data,
    col0: tl.constexpr, width: tl.constexpr, actual_dim: tl.constexpr,
):
    """Exclude both trailing rows and padded columns from the write mask."""
    cols = col0 + tl.arange(0, width)
    mask = row_ok[:, None] & (cols[None, :] < actual_dim)
    ptrs = dst + row_ids[:, None] * actual_dim + cols[None, :]
    tl.store(ptrs, data, mask)

@triton.jit
def _stage_dense_kernel(
    q_src, k_src, v_src, q_dst, k_dst, v_dst,
    sqh: tl.constexpr, sqr: tl.constexpr, sqd: tl.constexpr,
    skh: tl.constexpr, skr: tl.constexpr, skd: tl.constexpr,
    svh: tl.constexpr, svr: tl.constexpr, svd: tl.constexpr,
    seq: tl.constexpr, dim: tl.constexpr, numel: tl.constexpr,
    q_dense: tl.constexpr, k_dense: tl.constexpr, v_dense: tl.constexpr,
    copy_v: tl.constexpr, block: tl.constexpr,
):
    """Copy flat blocks, decoding source coordinates only for strided operands."""
    offset = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
    valid = offset < numel
    if not (q_dense and k_dense and v_dense):
        # These independent integer coordinates do not change pointer association.
        col = offset % dim
        row = offset // dim % seq
        head = offset // (seq * dim)
    if q_dense:
        q_ptrs = q_src + offset
    else:
        q_ptrs = q_src + head * sqh + row * sqr + col * sqd
    q_data = tl.load(q_ptrs, valid, other=0)
    if k_dense:
        k_ptrs = k_src + offset
    else:
        k_ptrs = k_src + head * skh + row * skr + col * skd
    k_data = tl.load(k_ptrs, valid, other=0)
    tl.store(q_dst + offset, q_data, valid)
    tl.store(k_dst + offset, k_data, valid)
    if copy_v:
        if v_dense:
            v_ptrs = v_src + offset
        else:
            v_ptrs = v_src + head * svh + row * svr + col * svd
        v_data = tl.load(v_ptrs, valid, other=0)
        tl.store(v_dst + offset, v_data, valid)

@triton.jit
def _stage_padded_kernel(
    q_src, k_src, v_src, q_dst, k_dst, v_dst,
    sqh: tl.constexpr, sqr: tl.constexpr, sqd: tl.constexpr,
    skh: tl.constexpr, skr: tl.constexpr, skd: tl.constexpr,
    svh: tl.constexpr, svr: tl.constexpr, svd: tl.constexpr,
    seq: tl.constexpr, dim: tl.constexpr, pitch: tl.constexpr,
    col_block: tl.constexpr, row_block: tl.constexpr, copy_v: tl.constexpr,
):
    """Pack strided Q/K/V into half storage with explicit zero-filled padding."""
    col = tl.arange(0, col_block)
    row = tl.program_id(0) * row_block + tl.arange(0, row_block)
    head = tl.program_id(1).to(tl.int64)
    row_ok = row[:, None] < seq
    read_ok = row_ok & (col[None, :] < dim)
    write_ok = row_ok & (col[None, :] < pitch)
    dst_offset = (head * seq + row[:, None]) * pitch + col[None, :]

    q_ptrs = q_src + head * sqh + row[:, None] * sqr + col[None, :] * sqd
    q_data = tl.load(q_ptrs, read_ok, other=0)
    tl.store(q_dst + dst_offset, q_data.to(tl.float16), write_ok)

    k_ptrs = k_src + head * skh + row[:, None] * skr + col[None, :] * skd
    k_data = tl.load(k_ptrs, read_ok, other=0)
    tl.store(k_dst + dst_offset, k_data.to(tl.float16), write_ok)

    if copy_v:
        v_ptrs = v_src + head * svh + row[:, None] * svr + col[None, :] * svd
        v_data = tl.load(v_ptrs, read_ok, other=0)
        tl.store(v_dst + dst_offset, v_data.to(tl.float16), write_ok)

@triton.jit
def _update_output_tile(
    output_accumulator_0,
    output_accumulator_1,
    output_accumulator_2,
    running_maximum,
    running_mass,
    query_slice_0,
    query_slice_1,
    query_slice_2,
    key_base,
    value_base,
    key_indices,
    sequence_length: tl.constexpr,
    padded_features: tl.constexpr,
    first_width: tl.constexpr,
    second_width: tl.constexpr,
    third_width: tl.constexpr,
    score_scale: tl.constexpr,
    mass_limit: tl.constexpr,
    check_bounds: tl.constexpr,
    defer_rescale: tl.constexpr,
):
    """Advance the output numerator and softmax state by one key tile."""
    key_bounds = key_indices < sequence_length
    key_slice_0 = _read_feature_major(
        key_base, key_indices, key_bounds, 0, first_width, padded_features, check_bounds
    )
    score_product = tl.dot(query_slice_0, key_slice_0)
    if second_width > 0:
        key_slice_1 = _read_feature_major(
            key_base,
            key_indices,
            key_bounds,
            first_width,
            second_width,
            padded_features,
            check_bounds,
        )
        score_product = tl.dot(query_slice_1, key_slice_1, score_product)
    if third_width > 0:
        key_slice_2 = _read_feature_major(
            key_base,
            key_indices,
            key_bounds,
            first_width + second_width,
            third_width,
            padded_features,
            check_bounds,
        )
        score_product = tl.dot(query_slice_2, key_slice_2, score_product)
    if check_bounds:
        score_product = tl.where(key_bounds[None, :], score_product, -float("inf"))
    if defer_rescale:
        probabilities = tl.math.exp2(
            score_product * score_scale - running_maximum[:, None]
        )
        tile_mass = tl.sum(probabilities, 1)
        if tl.max(tile_mass, 0) > mass_limit:
            next_maximum = tl.maximum(
                running_maximum, tl.max(score_product, 1) * score_scale
            )
            rescale_factor = tl.math.exp2(running_maximum - next_maximum)
            output_accumulator_0 = output_accumulator_0 * rescale_factor[:, None]
            if second_width > 0:
                output_accumulator_1 = output_accumulator_1 * rescale_factor[:, None]
            if third_width > 0:
                output_accumulator_2 = output_accumulator_2 * rescale_factor[:, None]
            running_mass = running_mass * rescale_factor
            running_maximum = next_maximum
            probabilities = tl.math.exp2(
                score_product * score_scale - running_maximum[:, None]
            )
            tile_mass = tl.sum(probabilities, 1)
    else:
        next_maximum = tl.maximum(
            running_maximum, tl.max(score_product, 1) * score_scale
        )
        rescale_factor = tl.math.exp2(running_maximum - next_maximum)
        output_accumulator_0 = output_accumulator_0 * rescale_factor[:, None]
        if second_width > 0:
            output_accumulator_1 = output_accumulator_1 * rescale_factor[:, None]
        if third_width > 0:
            output_accumulator_2 = output_accumulator_2 * rescale_factor[:, None]
        running_mass = running_mass * rescale_factor
        running_maximum = next_maximum
        probabilities = tl.math.exp2(
            score_product * score_scale - running_maximum[:, None]
        )
        tile_mass = tl.sum(probabilities, 1)
    running_mass = running_mass + tile_mass
    half_probabilities = probabilities.to(tl.float16)
    value_slice_0 = _read_token_major(
        value_base,
        key_indices,
        key_bounds,
        0,
        first_width,
        padded_features,
        check_bounds,
    )
    output_accumulator_0 = tl.dot(
        half_probabilities, value_slice_0, output_accumulator_0
    )
    if second_width > 0:
        value_slice_1 = _read_token_major(
            value_base,
            key_indices,
            key_bounds,
            first_width,
            second_width,
            padded_features,
            check_bounds,
        )
        output_accumulator_1 = tl.dot(
            half_probabilities, value_slice_1, output_accumulator_1
        )
    if third_width > 0:
        value_slice_2 = _read_token_major(
            value_base,
            key_indices,
            key_bounds,
            first_width + second_width,
            third_width,
            padded_features,
            check_bounds,
        )
        output_accumulator_2 = tl.dot(
            half_probabilities, value_slice_2, output_accumulator_2
        )
    return (
        output_accumulator_0,
        output_accumulator_1,
        output_accumulator_2,
        running_maximum,
        running_mass,
    )

@triton.jit
def _packed_forward_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    output_ptr,
    log_partition_ptr,
    output_sum_ptr,
    sequence_length: tl.constexpr,
    feature_count: tl.constexpr,
    padded_features: tl.constexpr,
    first_width: tl.constexpr,
    second_width: tl.constexpr,
    third_width: tl.constexpr,
    query_block: tl.constexpr,
    key_block: tl.constexpr,
    score_scale: tl.constexpr,
    mass_limit: tl.constexpr,
):
    """Produce output, log partition and rounded-output row sums."""
    head_index = tl.program_id(1).to(tl.int64)
    row_indices = tl.program_id(0) * query_block + tl.arange(0, query_block)
    valid_rows = row_indices < sequence_length
    head_base = head_index * sequence_length * padded_features
    query_slice_0 = _read_token_major(
        query_ptr + head_base,
        row_indices,
        valid_rows,
        0,
        first_width,
        padded_features,
        True,
    )
    output_accumulator_0 = tl.zeros([query_block, first_width], tl.float32)
    if second_width > 0:
        query_slice_1 = _read_token_major(
            query_ptr + head_base,
            row_indices,
            valid_rows,
            first_width,
            second_width,
            padded_features,
            True,
        )
        output_accumulator_1 = tl.zeros([query_block, second_width], tl.float32)
    else:
        query_slice_1 = tl.zeros([1], tl.float16)
        output_accumulator_1 = tl.zeros([1], tl.float32)
    if third_width > 0:
        query_slice_2 = _read_token_major(
            query_ptr + head_base,
            row_indices,
            valid_rows,
            first_width + second_width,
            third_width,
            padded_features,
            True,
        )
        output_accumulator_2 = tl.zeros([query_block, third_width], tl.float32)
    else:
        query_slice_2 = tl.zeros([1], tl.float16)
        output_accumulator_2 = tl.zeros([1], tl.float32)
    running_maximum = tl.full([query_block], -float("inf"), tl.float32)
    running_mass = tl.zeros([query_block], tl.float32)
    key_base = key_ptr + head_base
    value_base = value_ptr + head_base
    column_indices = tl.arange(0, key_block)
    complete_tiles: tl.constexpr = sequence_length // key_block
    if complete_tiles > 0:
        (
            output_accumulator_0,
            output_accumulator_1,
            output_accumulator_2,
            running_maximum,
            running_mass,
        ) = _update_output_tile(
            output_accumulator_0,
            output_accumulator_1,
            output_accumulator_2,
            running_maximum,
            running_mass,
            query_slice_0,
            query_slice_1,
            query_slice_2,
            key_base,
            value_base,
            column_indices,
            sequence_length,
            padded_features,
            first_width,
            second_width,
            third_width,
            score_scale,
            mass_limit,
            False,
            False,
        )
        for tile_index in range(1, complete_tiles):
            (
                output_accumulator_0,
                output_accumulator_1,
                output_accumulator_2,
                running_maximum,
                running_mass,
            ) = _update_output_tile(
                output_accumulator_0,
                output_accumulator_1,
                output_accumulator_2,
                running_maximum,
                running_mass,
                query_slice_0,
                query_slice_1,
                query_slice_2,
                key_base,
                value_base,
                tile_index * key_block + column_indices,
                sequence_length,
                padded_features,
                first_width,
                second_width,
                third_width,
                score_scale,
                mass_limit,
                False,
                True,
            )
    if complete_tiles * key_block < sequence_length:
        (
            output_accumulator_0,
            output_accumulator_1,
            output_accumulator_2,
            running_maximum,
            running_mass,
        ) = _update_output_tile(
            output_accumulator_0,
            output_accumulator_1,
            output_accumulator_2,
            running_maximum,
            running_mass,
            query_slice_0,
            query_slice_1,
            query_slice_2,
            key_base,
            value_base,
            complete_tiles * key_block + column_indices,
            sequence_length,
            padded_features,
            first_width,
            second_width,
            third_width,
            score_scale,
            mass_limit,
            True,
            False,
        )
    inverse_mass = 1.0 / running_mass
    destination_ptr = output_ptr + head_index * sequence_length * feature_count
    result_slice_0 = (output_accumulator_0 * inverse_mass[:, None]).to(
        output_ptr.dtype.element_ty
    )
    output_row_sum = tl.sum(result_slice_0.to(tl.float32), 1)
    _write_feature_slice(
        destination_ptr,
        row_indices,
        valid_rows,
        result_slice_0,
        0,
        first_width,
        feature_count,
    )
    if second_width > 0:
        result_slice_1 = (output_accumulator_1 * inverse_mass[:, None]).to(
            output_ptr.dtype.element_ty
        )
        output_row_sum += tl.sum(result_slice_1.to(tl.float32), 1)
        _write_feature_slice(
            destination_ptr,
            row_indices,
            valid_rows,
            result_slice_1,
            first_width,
            second_width,
            feature_count,
        )
    if third_width > 0:
        result_slice_2 = (output_accumulator_2 * inverse_mass[:, None]).to(
            output_ptr.dtype.element_ty
        )
        output_row_sum += tl.sum(result_slice_2.to(tl.float32), 1)
        _write_feature_slice(
            destination_ptr,
            row_indices,
            valid_rows,
            result_slice_2,
            first_width + second_width,
            third_width,
            feature_count,
        )
    tl.store(
        output_sum_ptr + head_index * sequence_length + row_indices,
        output_row_sum,
        valid_rows,
    )
    tl.store(
        log_partition_ptr + head_index * sequence_length + row_indices,
        running_maximum + tl.math.log2(running_mass),
        valid_rows,
    )

@triton.jit
def _generic_forward_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    output_ptr,
    log_partition_ptr,
    output_row_sum_ptr,
    query_head_stride: tl.constexpr,
    query_row_stride: tl.constexpr,
    query_feature_stride: tl.constexpr,
    key_head_stride: tl.constexpr,
    key_row_stride: tl.constexpr,
    key_feature_stride: tl.constexpr,
    value_head_stride: tl.constexpr,
    value_row_stride: tl.constexpr,
    value_feature_stride: tl.constexpr,
    sequence_length: tl.constexpr,
    feature_count: tl.constexpr,
    feature_block: tl.constexpr,
    query_block: tl.constexpr,
    key_block: tl.constexpr,
    score_scale: tl.constexpr,
    precision_mode: tl.constexpr,
):
    """Stream key/value tiles for the wide-head fallback."""
    head_index = tl.program_id(1).to(tl.int64)
    row_indices = tl.program_id(0) * query_block + tl.arange(0, query_block)
    column_indices = tl.arange(0, key_block)
    feature_indices = tl.arange(0, feature_block)
    query = tl.load(
        query_ptr
        + head_index * query_head_stride
        + row_indices[:, None] * query_row_stride
        + feature_indices[None, :] * query_feature_stride,
        (row_indices[:, None] < sequence_length)
        & (feature_indices[None, :] < feature_count),
        other=0,
    )
    running_maximum = tl.full((query_block,), -float("inf"), tl.float32)
    normalizing_mass = tl.full((query_block,), 0, tl.float32)
    accumulator = tl.full((query_block, feature_block), 0, tl.float32)
    for tile_index in range(tl.cdiv(sequence_length, key_block)):
        key_indices = tile_index * key_block + column_indices
        key = tl.load(
            key_ptr
            + head_index * key_head_stride
            + feature_indices[:, None] * key_feature_stride
            + key_indices[None, :] * key_row_stride,
            (feature_indices[:, None] < feature_count)
            & (key_indices[None, :] < sequence_length),
            other=0,
        )
        value = tl.load(
            value_ptr
            + head_index * value_head_stride
            + key_indices[:, None] * value_row_stride
            + feature_indices[None, :] * value_feature_stride,
            (key_indices[:, None] < sequence_length)
            & (feature_indices[None, :] < feature_count),
            other=0,
        )
        score_tile = _operand_product(query, key, precision_mode) * score_scale
        score_tile = tl.where(
            key_indices[None, :] < sequence_length, score_tile, -float("inf")
        )
        updated_maximum = tl.maximum(running_maximum, tl.max(score_tile, 1))
        rescale_factor = tl.exp2(running_maximum - updated_maximum)
        probabilities = tl.exp2(score_tile - updated_maximum[:, None])
        accumulator = _operand_product(
            probabilities, value, precision_mode, accumulator * rescale_factor[:, None]
        )
        normalizing_mass = normalizing_mass * rescale_factor + tl.sum(probabilities, 1)
        running_maximum = updated_maximum
    stored_output = (accumulator / normalizing_mass[:, None]).to(
        output_ptr.dtype.element_ty
    )
    tl.store(
        output_row_sum_ptr + head_index * sequence_length + row_indices,
        tl.sum(stored_output.to(tl.float32), 1),
        row_indices < sequence_length,
    )
    tl.store(
        output_ptr
        + (head_index * sequence_length + row_indices[:, None]) * feature_count
        + feature_indices[None, :],
        stored_output,
        (row_indices[:, None] < sequence_length)
        & (feature_indices[None, :] < feature_count),
    )
    tl.store(
        log_partition_ptr + head_index * sequence_length + row_indices,
        running_maximum + tl.log2(normalizing_mass),
        row_indices < sequence_length,
    )

@triton.jit
def _prepare_gradient_rows(
    output_ptr,
    upstream_ptr,
    value_ptr,
    row_contraction_ptr,
    value_reduction_ptr,
    value_head_stride: tl.constexpr,
    value_row_stride: tl.constexpr,
    value_feature_stride: tl.constexpr,
    upstream_head_stride: tl.constexpr,
    upstream_row_stride: tl.constexpr,
    upstream_feature_stride: tl.constexpr,
    sequence_length: tl.constexpr,
    feature_count: tl.constexpr,
    feature_block: tl.constexpr,
    query_block: tl.constexpr,
):
    """Prepare row contractions and optional sequence-broadcast reductions."""
    head_index = tl.program_id(1).to(tl.int64)
    row_indices = tl.program_id(0) * query_block + tl.arange(0, query_block)
    feature_indices = tl.arange(0, feature_block)
    valid_elements = (row_indices[:, None] < sequence_length) & (
        feature_indices[None, :] < feature_count
    )
    upstream_tile = tl.load(
        upstream_ptr
        + head_index * upstream_head_stride
        + row_indices[:, None] * upstream_row_stride
        + feature_indices[None, :] * upstream_feature_stride,
        valid_elements,
        other=0,
    ).to(tl.float32)
    output = tl.load(
        output_ptr
        + (head_index * sequence_length + row_indices[:, None]) * feature_count
        + feature_indices[None, :],
        valid_elements,
        other=0,
    )
    tl.store(
        row_contraction_ptr + head_index * sequence_length + row_indices,
        tl.sum(output.to(tl.float32) * upstream_tile, 1),
        row_indices < sequence_length,
    )

@triton.jit
def _generic_query_gradient(
    query_ptr,
    key_ptr,
    value_ptr,
    upstream_ptr,
    log_partition_ptr,
    row_contraction_ptr,
    value_reduction_ptr,
    query_gradient_ptr,
    query_head_stride: tl.constexpr,
    query_row_stride: tl.constexpr,
    query_feature_stride: tl.constexpr,
    key_head_stride: tl.constexpr,
    key_row_stride: tl.constexpr,
    key_feature_stride: tl.constexpr,
    value_head_stride: tl.constexpr,
    value_row_stride: tl.constexpr,
    value_feature_stride: tl.constexpr,
    upstream_head_stride: tl.constexpr,
    upstream_row_stride: tl.constexpr,
    upstream_feature_stride: tl.constexpr,
    sequence_length: tl.constexpr,
    feature_count: tl.constexpr,
    feature_block: tl.constexpr,
    query_block: tl.constexpr,
    key_block: tl.constexpr,
    score_scale: tl.constexpr,
    attention_scale: tl.constexpr,
    precision_mode: tl.constexpr,
):
    """Compute query derivatives for a general upstream gradient."""
    head_index = tl.program_id(1).to(tl.int64)
    row_indices = tl.program_id(0) * query_block + tl.arange(0, query_block)
    column_indices = tl.arange(0, key_block)
    feature_indices = tl.arange(0, feature_block)
    valid_elements = (row_indices[:, None] < sequence_length) & (
        feature_indices[None, :] < feature_count
    )
    query = tl.load(
        query_ptr
        + head_index * query_head_stride
        + row_indices[:, None] * query_row_stride
        + feature_indices[None, :] * query_feature_stride,
        valid_elements,
        other=0,
    )
    upstream_tile = tl.load(
        upstream_ptr
        + head_index * upstream_head_stride
        + row_indices[:, None] * upstream_row_stride
        + feature_indices[None, :] * upstream_feature_stride,
        valid_elements,
        other=0,
    )
    log_partition = tl.load(
        log_partition_ptr + head_index * sequence_length + row_indices,
        row_indices < sequence_length,
        other=0,
    )
    row_contraction = tl.load(
        row_contraction_ptr + head_index * sequence_length + row_indices,
        row_indices < sequence_length,
        other=0,
    )
    accumulator = tl.full((query_block, feature_block), 0, tl.float32)
    probability_gain: tl.constexpr = 1024.0 if precision_mode == "fp16" else 1.0
    for tile_index in range(tl.cdiv(sequence_length, key_block)):
        key_indices = tile_index * key_block + column_indices
        valid_transposed_elements = (feature_indices[:, None] < feature_count) & (
            key_indices[None, :] < sequence_length
        )
        transposed_keys = tl.load(
            key_ptr
            + head_index * key_head_stride
            + feature_indices[:, None] * key_feature_stride
            + key_indices[None, :] * key_row_stride,
            valid_transposed_elements,
            other=0,
        )
        score_tile = (
            _operand_product(query, transposed_keys, precision_mode) * score_scale
        )
        probabilities = tl.exp2(score_tile - log_partition[:, None])
        probabilities = tl.where(
            key_indices[None, :] < sequence_length, probabilities, 0
        )
        transposed_values = tl.load(
            value_ptr
            + head_index * value_head_stride
            + feature_indices[:, None] * value_feature_stride
            + key_indices[None, :] * value_row_stride,
            valid_transposed_elements,
            other=0,
        )
        probability_gradient = _operand_product(
            upstream_tile, transposed_values, precision_mode
        )
        score_gradient = (
            probabilities
            * (probability_gradient - row_contraction[:, None])
            * probability_gain
        )
        accumulator = _operand_product(
            score_gradient, tl.trans(transposed_keys), precision_mode, accumulator
        )
    tl.store(
        query_gradient_ptr
        + (head_index * sequence_length + row_indices[:, None]) * feature_count
        + feature_indices[None, :],
        accumulator * (attention_scale / probability_gain),
        valid_elements,
    )

@triton.jit
def _generic_key_value_gradient(
    query_ptr,
    key_ptr,
    value_ptr,
    upstream_ptr,
    log_partition_ptr,
    row_contraction_ptr,
    value_reduction_ptr,
    key_gradient_ptr,
    value_gradient_ptr,
    query_head_stride: tl.constexpr,
    query_row_stride: tl.constexpr,
    query_feature_stride: tl.constexpr,
    key_head_stride: tl.constexpr,
    key_row_stride: tl.constexpr,
    key_feature_stride: tl.constexpr,
    value_head_stride: tl.constexpr,
    value_row_stride: tl.constexpr,
    value_feature_stride: tl.constexpr,
    upstream_head_stride: tl.constexpr,
    upstream_row_stride: tl.constexpr,
    upstream_feature_stride: tl.constexpr,
    sequence_length: tl.constexpr,
    feature_count: tl.constexpr,
    feature_block: tl.constexpr,
    query_block: tl.constexpr,
    key_block: tl.constexpr,
    score_scale: tl.constexpr,
    attention_scale: tl.constexpr,
    precision_mode: tl.constexpr,
):
    """Compute key/value derivatives without an attention-sized workspace."""
    head_index = tl.program_id(1).to(tl.int64)
    key_indices = tl.program_id(0) * key_block + tl.arange(0, key_block)
    row_indices = tl.arange(0, query_block)
    feature_indices = tl.arange(0, feature_block)
    valid_elements = (key_indices[:, None] < sequence_length) & (
        feature_indices[None, :] < feature_count
    )
    key = tl.load(
        key_ptr
        + head_index * key_head_stride
        + key_indices[:, None] * key_row_stride
        + feature_indices[None, :] * key_feature_stride,
        valid_elements,
        other=0,
    )
    key_accumulator = tl.full((key_block, feature_block), 0, tl.float32)
    value = tl.load(
        value_ptr
        + head_index * value_head_stride
        + key_indices[:, None] * value_row_stride
        + feature_indices[None, :] * value_feature_stride,
        valid_elements,
        other=0,
    )
    value_accumulator = tl.full((key_block, feature_block), 0, tl.float32)
    probability_gain: tl.constexpr = 1024.0 if precision_mode == "fp16" else 1.0
    for tile_index in range(tl.cdiv(sequence_length, query_block)):
        query_indices = tile_index * query_block + row_indices
        valid_transposed_elements = (feature_indices[:, None] < feature_count) & (
            query_indices[None, :] < sequence_length
        )
        transposed_queries = tl.load(
            query_ptr
            + head_index * query_head_stride
            + feature_indices[:, None] * query_feature_stride
            + query_indices[None, :] * query_row_stride,
            valid_transposed_elements,
            other=0,
        )
        transposed_upstream = tl.load(
            upstream_ptr
            + head_index * upstream_head_stride
            + feature_indices[:, None] * upstream_feature_stride
            + query_indices[None, :] * upstream_row_stride,
            valid_transposed_elements,
            other=0,
        )
        log_partition = tl.load(
            log_partition_ptr + head_index * sequence_length + query_indices,
            query_indices < sequence_length,
            other=0,
        )
        row_contraction = tl.load(
            row_contraction_ptr + head_index * sequence_length + query_indices,
            query_indices < sequence_length,
            other=0,
        )
        score_tile = (
            _operand_product(key, transposed_queries, precision_mode) * score_scale
        )
        probabilities = tl.exp2(score_tile - log_partition[None, :])
        probabilities = tl.where(
            query_indices[None, :] < sequence_length, probabilities, 0
        )
        probability_gradient = _operand_product(
            value, transposed_upstream, precision_mode
        )
        score_gradient = (
            probabilities
            * (probability_gradient - row_contraction[None, :])
            * probability_gain
        )
        if precision_mode == "fp16":
            key_accumulator = _residual_product(
                score_gradient, tl.trans(transposed_queries), key_accumulator
            )
        else:
            key_accumulator = _operand_product(
                score_gradient,
                tl.trans(transposed_queries),
                precision_mode,
                key_accumulator,
            )
        value_accumulator = _operand_product(
            probabilities * probability_gain,
            tl.trans(transposed_upstream),
            precision_mode,
            value_accumulator,
        )
    tl.store(
        key_gradient_ptr
        + (head_index * sequence_length + key_indices[:, None]) * feature_count
        + feature_indices[None, :],
        key_accumulator * (attention_scale / probability_gain),
        valid_elements,
    )
    value_gradient_tile = value_accumulator / probability_gain
    tl.store(
        value_gradient_ptr
        + (head_index * sequence_length + key_indices[:, None]) * feature_count
        + feature_indices[None, :],
        value_gradient_tile,
        valid_elements,
    )

class _WideHeadExecutor:
    """Dispatch the original wide-head path without changing its storage policy."""

    INPUT_DTYPE = torch.float32
    DOT_MODE = "fp16"
    OUTPUT_DTYPE = torch.float16
    USE_MANAGED_OUTPUT = True
    NUM_WARPS = 4
    FP16_LAUNCHES = {
        128: ((128, 32, 4, 2), (64, 128, 4, 1), (64, 64, 4, 1)),
        100: ((128, 32, 4, 1), (64, 128, 4, 1), (64, 64, 4, 1)),
    }
    PACK_QKV = True
    PACK_HEADS = 8
    PACK_MIN_ELEMS = 1 << 20
    PACK_MIN_N = 4096

    @staticmethod
    def stage_head_group(query, key, value, staging_storage, include_values=True):
        head_count, token_count, head_width = query.shape
        staged_query, staged_key, staged_value = (
            item[:head_count] for item in staging_storage
        )
        dense_strides = (token_count * head_width, head_width, 1)
        _stage_dense_kernel[triton.cdiv(head_count * token_count * head_width, 4096),](
            query,
            key,
            value,
            staged_query,
            staged_key,
            staged_value,
            *query.stride(),
            *key.stride(),
            *value.stride(),
            token_count,
            head_width,
            head_count * token_count * head_width,
            query.stride() == dense_strides,
            key.stride() == dense_strides,
            value.stride() == dense_strides,
            include_values,
            4096,
            num_warps=8,
        )
        return (staged_query, staged_key, staged_value)

    @staticmethod
    def select_schedule(mode, dim, staged=True, seq=None):
        """Separate precision selection from the staging-dependent tile policy."""
        if mode == "fp16":
            if staged:
                cfg = _WideHeadExecutor.FP16_LAUNCHES.get(
                    dim, _WideHeadExecutor.FP16_LAUNCHES[128]
                )
                if dim == 128 and seq is not None and seq > 32768:
                    return cfg[0], cfg[1], (64, 64, 4, 2)
                return cfg
            kv_stages = 2 if dim == 100 else 1
            return (128, 32, 4, 2), (64, 64, 4, 1), (64, 64, 4, kv_stages)
        return (64, 32, 4, 2), (32, 32, 4, 2), (32, 32, 4, 2)

    @staticmethod
    def compute_output(autograd_state, query, key, value):
        head_count, token_count, head_width = query.shape
        operand_mode = _WideHeadExecutor.DOT_MODE
        result_dtype = _WideHeadExecutor.OUTPUT_DTYPE
        result_item_bytes = 2 if result_dtype == torch.float16 else 4
        use_staging = (
            _WideHeadExecutor.PACK_QKV
            and operand_mode == "fp16"
            and (query.dtype == torch.float32)
            and (token_count >= _WideHeadExecutor.PACK_MIN_N)
            and (query.numel() >= _WideHeadExecutor.PACK_MIN_ELEMS)
        )
        estimated_working_bytes = (
            query.numel() * (6 * query.element_size() + result_item_bytes)
            + 16 * head_count * token_count
        )
        device_capacity = torch.cuda.get_device_properties(query.device).total_memory
        use_managed_storage = (
            _WideHeadExecutor.USE_MANAGED_OUTPUT
            and estimated_working_bytes > 0.92 * device_capacity
        )
        autograd_state.reserved_gradients = None
        if use_managed_storage:
            with torch.cuda.device(query.device):
                autograd_state.reserved_gradients = tuple(
                    (
                        torch.empty(item.shape, device=item.device, dtype=item.dtype)
                        for item in (query, key, value)
                    )
                )
                output = _allocate_managed_storage(
                    (head_count, token_count, head_width), result_dtype, query.device
                )
        else:
            output = torch.empty(
                (head_count, token_count, head_width),
                device=query.device,
                dtype=result_dtype,
            )
        log_partition = torch.empty(
            (head_count, token_count), device=query.device, dtype=torch.float32
        )
        output_row_sum = torch.empty_like(log_partition)
        output_launch, query_launch, key_value_launch = (
            _WideHeadExecutor.select_schedule(
                operand_mode, head_width, use_staging, token_count
            )
        )
        query_tile_rows, key_tile_rows, output_warps, output_stages = output_launch
        with torch.cuda.device(query.device):
            head_batch_size = (
                min(head_count, _WideHeadExecutor.PACK_HEADS)
                if use_staging
                else head_count
            )
            if use_staging:
                staging_storage = tuple(
                    (
                        torch.empty(
                            (head_batch_size, token_count, head_width),
                            device=query.device,
                            dtype=torch.float16,
                        )
                        for unused_value in range(3)
                    )
                )
            for head_begin in range(0, head_count, head_batch_size):
                head_end = min(head_begin + head_batch_size, head_count)
                query_batch, key_batch, value_batch = (
                    query[head_begin:head_end],
                    key[head_begin:head_end],
                    value[head_begin:head_end],
                )
                if use_staging:
                    query_batch, key_batch, value_batch = (
                        _WideHeadExecutor.stage_head_group(
                            query_batch, key_batch, value_batch, staging_storage
                        )
                    )
                _generic_forward_kernel[
                    triton.cdiv(token_count, query_tile_rows), head_end - head_begin
                ](
                    query_batch,
                    key_batch,
                    value_batch,
                    output[head_begin:head_end],
                    log_partition[head_begin:head_end],
                    output_row_sum[head_begin:head_end],
                    *query_batch.stride(),
                    *key_batch.stride(),
                    *value_batch.stride(),
                    token_count,
                    head_width,
                    triton.next_power_of_2(head_width),
                    query_tile_rows,
                    key_tile_rows,
                    head_width ** (-0.5) * 1.4426950408889634,
                    operand_mode,
                    num_warps=output_warps,
                    num_stages=output_stages,
                )
        autograd_state.save_for_backward(
            query, key, value, output, log_partition, output_row_sum
        )
        autograd_state.operand_precision = operand_mode
        autograd_state.wide_query_schedule, autograd_state.wide_key_value_schedule = (
            query_launch,
            key_value_launch,
        )
        autograd_state.uses_dense_staging, autograd_state.wide_head_batch = (
            use_staging,
            head_batch_size,
        )
        return output

    @staticmethod
    def compute_gradients(autograd_state, upstream_gradient):
        query, key, value, output, log_partition, output_row_sum = (
            autograd_state.saved_tensors
        )
        head_count, token_count, head_width = query.shape
        if autograd_state.reserved_gradients is not None:
            query_gradient, key_gradient, value_gradient = (
                autograd_state.reserved_gradients
            )
            del autograd_state.reserved_gradients
        else:
            query_gradient = torch.empty(
                query.shape, device=query.device, dtype=query.dtype
            )
            key_gradient = torch.empty(key.shape, device=key.device, dtype=key.dtype)
            value_gradient = torch.empty(
                value.shape, device=value.device, dtype=value.dtype
            )
        broadcast_upstream = (
            upstream_gradient.stride(1) == 0 and upstream_gradient.stride(2) == 0
        )
        row_contraction = (
            output_row_sum
            if broadcast_upstream
            else torch.empty(
                (head_count, token_count), device=query.device, dtype=torch.float32
            )
        )
        value_reduction = torch.empty_like(row_contraction)
        query_launch, key_value_launch = (
            autograd_state.wide_query_schedule,
            autograd_state.wide_key_value_schedule,
        )
        if upstream_gradient.stride(1) != 0:
            unused_value, query_launch, key_value_launch = (
                _WideHeadExecutor.select_schedule(
                    autograd_state.operand_precision, head_width, False, token_count
                )
            )
        feature_tile_width = triton.next_power_of_2(head_width)
        with torch.cuda.device(query.device):
            _prepare_gradient_rows[triton.cdiv(token_count, 64), head_count](
                output,
                upstream_gradient,
                value,
                row_contraction,
                value_reduction,
                *value.stride(),
                *upstream_gradient.stride(),
                token_count,
                head_width,
                feature_tile_width,
                64,
                num_warps=4,
            )
            if autograd_state.uses_dense_staging:
                staging_storage = tuple(
                    (
                        torch.empty(
                            (autograd_state.wide_head_batch, token_count, head_width),
                            device=query.device,
                            dtype=torch.float16,
                        )
                        for unused_value in range(3)
                    )
                )
            for head_begin in range(0, head_count, autograd_state.wide_head_batch):
                head_end = min(head_begin + autograd_state.wide_head_batch, head_count)
                query_batch, key_batch, value_batch = (
                    query[head_begin:head_end],
                    key[head_begin:head_end],
                    value[head_begin:head_end],
                )
                upstream_batch = upstream_gradient[head_begin:head_end]
                if autograd_state.uses_dense_staging:
                    query_batch, key_batch, value_batch = (
                        _WideHeadExecutor.stage_head_group(
                            query_batch,
                            key_batch,
                            value_batch,
                            staging_storage,
                            upstream_gradient.stride(1) != 0,
                        )
                    )
                shared_arguments = (
                    *query_batch.stride(),
                    *key_batch.stride(),
                    *value_batch.stride(),
                    *upstream_batch.stride(),
                    token_count,
                    head_width,
                    feature_tile_width,
                )
                query_tile_rows, key_tile_rows, warp_count, pipeline_depth = (
                    query_launch
                )
                _generic_query_gradient[
                    triton.cdiv(token_count, query_tile_rows), head_end - head_begin
                ](
                    query_batch,
                    key_batch,
                    value_batch,
                    upstream_batch,
                    log_partition[head_begin:head_end],
                    row_contraction[head_begin:head_end],
                    value_reduction[head_begin:head_end],
                    query_gradient[head_begin:head_end],
                    *shared_arguments,
                    query_tile_rows,
                    key_tile_rows,
                    head_width ** (-0.5) * 1.4426950408889634,
                    head_width ** (-0.5),
                    autograd_state.operand_precision,
                    num_warps=warp_count,
                    num_stages=pipeline_depth,
                )
                query_tile_rows, key_tile_rows, warp_count, pipeline_depth = (
                    key_value_launch
                )
                _generic_key_value_gradient[
                    triton.cdiv(token_count, key_tile_rows), head_end - head_begin
                ](
                    query_batch,
                    key_batch,
                    value_batch,
                    upstream_batch,
                    log_partition[head_begin:head_end],
                    row_contraction[head_begin:head_end],
                    value_reduction[head_begin:head_end],
                    key_gradient[head_begin:head_end],
                    value_gradient[head_begin:head_end],
                    *shared_arguments,
                    query_tile_rows,
                    key_tile_rows,
                    head_width ** (-0.5) * 1.4426950408889634,
                    head_width ** (-0.5),
                    autograd_state.operand_precision,
                    num_warps=warp_count,
                    num_stages=pipeline_depth,
                )
        return (query_gradient, key_gradient, value_gradient)

class FlashAttention2(torch.autograd.Function):
    """Evaluator-compatible attention for equal-shaped CUDA [H, N, D] inputs.

    Configuration names remain compatible. Selected launch schedules are
    tuning candidates rather than validated improvements.
    Both broadcast and general upstream gradients produce all three derivatives.
    Inputs must use matching FP32/FP16/BF16 dtypes, H/N > 0 and 1 <= D <= 256;
    these preconditions are not checked here.
    """

    INPUT_DTYPE = torch.float32
    OUTPUT_DTYPE = torch.float16
    DOT_MODE = "fp16"
    USE_MANAGED_OUTPUT = True
    MANAGED_TENSOR = "o"
    HOST_ADVISE = False
    HOST_POOL = False
    PACK_HEADS = 8
    PACK_BYTES = 2**30
    SPLIT_D = True
    LIMIT = 256.0
    FAST_FWD = (128, 32, 4, 2)
    FAST_FWD_SHORT = (128, 32, 4, 3)
    FAST_DQ = (128, 32, 4, 3)
    FAST_DKV = (64, 128, 8, 3, 1)
    LONG_N = 32768
    # Keep tile geometry fixed while trying shallower dQ pipelines at DP=128.
    FAST_TABLE = {
        (112, True): ((128, 32, 4, 2), (128, 32, 4, 3), (32, 128, 4, 4, 1)),
        (128, False): ((128, 32, 4, 3), (128, 32, 4, 3), (64, 128, 8, 3, 1)),
        (128, True): ((128, 32, 4, 2), (128, 32, 4, 2), (64, 128, 8, 3, 1)),
    }
    DKV_WIDEST = 64

    @staticmethod
    def packed_layout(dim):
        """Choose storage width separately from the two MMA decompositions."""
        if FlashAttention2.SPLIT_D:
            padded = -(-dim // 16) * 16
        else:
            padded = max(16, triton.next_power_of_2(dim))
        out_parts = _split_dim(padded)
        grad_parts = _split_dim(padded, FlashAttention2.DKV_WIDEST)
        return padded, out_parts, grad_parts

    @staticmethod
    def kernel_schedule(padded, seq):
        """Use a shape-specific schedule when present, otherwise use the defaults."""
        impl = FlashAttention2
        long_seq = seq > impl.LONG_N
        if long_seq:
            fwd = impl.FAST_FWD
        else:
            fwd = impl.FAST_FWD_SHORT
        fallback = fwd, impl.FAST_DQ, impl.FAST_DKV
        return impl.FAST_TABLE.get((padded, long_seq), fallback)

    @staticmethod
    def stage_operands(
        query, key, value, staging_storage, padded_width, include_values=True
    ):
        staged_head_count, token_count, head_width = query.shape
        staged_query, staged_key, staged_value = (
            item[:staged_head_count] for item in staging_storage
        )
        row_indices = 32
        _stage_padded_kernel[triton.cdiv(token_count, row_indices), staged_head_count](
            query,
            key,
            value,
            staged_query,
            staged_key,
            staged_value,
            *query.stride(),
            *key.stride(),
            *value.stride(),
            token_count,
            head_width,
            padded_width,
            triton.next_power_of_2(padded_width),
            row_indices,
            include_values,
            num_warps=4,
        )
        return (staged_query, staged_key, staged_value)

    @staticmethod
    def forward(autograd_state, query, key, value):
        head_count, token_count, head_width = query.shape
        # Kernels address all three operands using the query shape and device.
        if (
            key.shape != query.shape
            or value.shape != query.shape
            or key.device != query.device
            or value.device != query.device
        ):
            raise ValueError("Q, K and V must share shape and device")
        autograd_state.uses_padded_operands = head_width <= 128
        if not autograd_state.uses_padded_operands:
            return _WideHeadExecutor.compute_output(autograd_state, query, key, value)
        return FlashAttention2._forward_packed(
            autograd_state, query, key, value, head_count, token_count, head_width
        )

    @staticmethod
    def _forward_packed(
        autograd_state, query, key, value, head_count, token_count, head_width
    ):
        padded_width, output_slices, gradient_slices = FlashAttention2.packed_layout(
            head_width
        )
        output_schedule, query_schedule, key_value_schedule = (
            FlashAttention2.kernel_schedule(padded_width, token_count)
        )
        result_dtype = FlashAttention2.OUTPUT_DTYPE
        result_item_bytes = 2 if result_dtype == torch.float16 else 4
        estimated_working_bytes = (
            query.numel() * (6 * query.element_size() + result_item_bytes)
            + 16 * head_count * token_count
        )
        device_capacity = torch.cuda.get_device_properties(query.device).total_memory
        use_managed_storage = (
            FlashAttention2.USE_MANAGED_OUTPUT
            and estimated_working_bytes > 0.92 * device_capacity
        )
        autograd_state.reserved_gradients = None
        with torch.cuda.device(query.device):
            if use_managed_storage:
                manage_value_gradient = (
                    FlashAttention2.MANAGED_TENSOR == "dv"
                    and value.dtype in (torch.float16, torch.float32)
                )
                prefer_host, retain_allocation = (
                    FlashAttention2.HOST_ADVISE,
                    FlashAttention2.HOST_POOL,
                )
                if manage_value_gradient:
                    output = torch.empty(
                        (head_count, token_count, head_width),
                        device=query.device,
                        dtype=result_dtype,
                    )
                    autograd_state.reserved_gradients = (
                        torch.empty(
                            query.shape, device=query.device, dtype=query.dtype
                        ),
                        torch.empty(key.shape, device=key.device, dtype=key.dtype),
                        _allocate_managed_storage(
                            tuple(value.shape),
                            value.dtype,
                            value.device,
                            prefer_host,
                            retain_allocation,
                        ),
                    )
                else:
                    autograd_state.reserved_gradients = tuple(
                        (
                            torch.empty(
                                item.shape, device=item.device, dtype=item.dtype
                            )
                            for item in (query, key, value)
                        )
                    )
                    output = _allocate_managed_storage(
                        (head_count, token_count, head_width),
                        result_dtype,
                        query.device,
                        prefer_host,
                        retain_allocation,
                    )
            else:
                output = torch.empty(
                    (head_count, token_count, head_width),
                    device=query.device,
                    dtype=result_dtype,
                )
            log_partition = torch.empty(
                (head_count, token_count), device=query.device, dtype=torch.float32
            )
            output_row_sum = torch.empty_like(log_partition)
            head_batch_count = min(head_count, FlashAttention2.PACK_HEADS)
            if (
                not use_managed_storage
                and 6 * head_count * token_count * padded_width
                <= FlashAttention2.PACK_BYTES
            ):
                head_batch_count = head_count
            staging_storage = tuple(
                (
                    torch.empty(
                        (head_batch_count, token_count, padded_width),
                        device=query.device,
                        dtype=torch.float16,
                    )
                    for unused_value in range(3)
                )
            )
            query_tile_rows, key_tile_rows, warp_count, pipeline_depth = output_schedule
            scaled_log2_attention = head_width ** (-0.5) * 1.4426950408889634
            for head_begin in range(0, head_count, head_batch_count):
                head_end = min(head_begin + head_batch_count, head_count)
                staged_query, staged_key, staged_value = FlashAttention2.stage_operands(
                    query[head_begin:head_end],
                    key[head_begin:head_end],
                    value[head_begin:head_end],
                    staging_storage,
                    padded_width,
                )
                _packed_forward_kernel[
                    triton.cdiv(token_count, query_tile_rows), head_end - head_begin
                ](
                    staged_query,
                    staged_key,
                    staged_value,
                    output[head_begin:head_end],
                    log_partition[head_begin:head_end],
                    output_row_sum[head_begin:head_end],
                    token_count,
                    head_width,
                    padded_width,
                    *output_slices,
                    query_tile_rows,
                    key_tile_rows,
                    scaled_log2_attention,
                    FlashAttention2.LIMIT,
                    num_warps=warp_count,
                    num_stages=pipeline_depth,
                )
            del staging_storage
        autograd_state.save_for_backward(
            query, key, value, output, log_partition, output_row_sum
        )
        (
            autograd_state.padded_feature_count,
            autograd_state.feature_slices,
            autograd_state.key_value_slices,
            autograd_state.heads_per_batch,
        ) = (padded_width, output_slices, gradient_slices, head_batch_count)
        autograd_state.query_schedule, autograd_state.key_value_schedule = (
            query_schedule,
            key_value_schedule,
        )
        return output

    @staticmethod
    def backward(autograd_state, upstream_gradient):
        if autograd_state.uses_padded_operands:
            query = autograd_state.saved_tensors[0]
            _, query_launch, key_value_launch = _WideHeadExecutor.select_schedule(
                "fp16", query.shape[-1], False, query.shape[1]
            )
            autograd_state.operand_precision = "fp16"
            autograd_state.wide_query_schedule = query_launch
            autograd_state.wide_key_value_schedule = key_value_launch
            autograd_state.uses_dense_staging = False
            autograd_state.wide_head_batch = autograd_state.heads_per_batch
        return _WideHeadExecutor.compute_gradients(
            autograd_state, upstream_gradient.contiguous()
        )
