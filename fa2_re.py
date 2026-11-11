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
    """Stage bounded head groups and keep large outputs in managed storage."""

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
        """Use conservative tiles before shape-specific launch tuning."""
        return (32, 32, 4, 1), (32, 32, 4, 1), (32, 32, 4, 1)

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
    """Memory-bounded attention with a general dense-gradient backward."""

    INPUT_DTYPE = torch.float32
    OUTPUT_DTYPE = torch.float16

    @staticmethod
    def forward(ctx, query, key, value):
        if query.shape != key.shape or query.shape != value.shape:
            raise ValueError("Q, K and V must share shape")
        if query.device != key.device or query.device != value.device:
            raise ValueError("Q, K and V must share device")
        return _WideHeadExecutor.compute_output(ctx, query, key, value)

    @staticmethod
    def backward(ctx, upstream):
        return _WideHeadExecutor.compute_gradients(ctx, upstream.contiguous())
