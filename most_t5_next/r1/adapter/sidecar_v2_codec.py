"""Safe binary codec for bounded PCQM geometry-sidecar v2 records.

The historical v1 smoke used ``pickle`` inside LMDB.  That is unsuitable for
an independently audited data release because merely inspecting a corrupted
value can execute code.  This codec writes a canonical UTF-8 JSON header plus
explicit, length-delimited raw C-order NumPy blocks.  It contains no generic
object deserialization and rejects any unknown payload structure.

The module is a producer/replay utility only.  A later independent reference
audit must implement its own minimal decoder rather than importing this file.
"""

from __future__ import print_function

import hashlib
import json
import struct


PAYLOAD_SCHEMA = "most-t5-r1/p1-pcqm-geometry-sidecar-payload/v2"
MAGIC = b"MST5PCQM2\x00"
HEADER_LENGTH_BYTES = 4
MAX_HEADER_BYTES = 16 * 1024 * 1024
MAX_PAYLOAD_BYTES = 16 * 1024 * 1024
MAX_ARRAY_BLOCKS = 100000
_PLACEHOLDER_KEYS = frozenset(("__array_block__", "dtype", "shape", "order", "sha256"))
_BLOCK_KEYS = frozenset(("index", "dtype", "shape", "order", "offset", "nbytes", "sha256"))
_DTYPE_BY_NAME = {
    "int32": "<i4",
    "float32": "<f4",
    "bool": "|b1",
}
_HEX64 = frozenset("0123456789abcdef")


def canonical_json_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX64


def _array_descriptor(np, value):
    if not isinstance(value, np.ndarray):
        raise TypeError("sidecar payload array must be a NumPy ndarray")
    if not value.flags.c_contiguous:
        raise ValueError("sidecar payload array must be C-contiguous")
    dtype_name = str(value.dtype)
    if dtype_name not in _DTYPE_BY_NAME:
        raise ValueError("unsupported sidecar payload dtype: {}".format(dtype_name))
    return {
        "dtype": dtype_name,
        "shape": [int(item) for item in value.shape],
        "order": "C",
        "sha256": sha256_bytes(value.tobytes(order="C")),
    }


def logical_projection(np, value):
    """Stable raw-free projection used for the record logical hash."""
    if isinstance(value, np.ndarray):
        return {"__ndarray__": _array_descriptor(np, value)}
    if isinstance(value, dict):
        return {str(key): logical_projection(np, value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [logical_projection(np, item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError("unsupported value in sidecar logical projection: {}".format(type(value).__name__))


def logical_record_sha256(np, record):
    return sha256_bytes(canonical_json_bytes(logical_projection(np, record)))


def _externalize_arrays(np, value, blocks):
    if isinstance(value, np.ndarray):
        descriptor = _array_descriptor(np, value)
        index = len(blocks)
        blocks.append(value)
        return {"__array_block__": index, **descriptor}
    if isinstance(value, dict):
        return {str(key): _externalize_arrays(np, value[key], blocks) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_externalize_arrays(np, item, blocks) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError("unsupported value in sidecar payload: {}".format(type(value).__name__))


def encode_record(np, record):
    """Encode one validated record without ever serializing Python objects."""
    blocks = []
    record_projection = _externalize_arrays(np, record, blocks)
    if len(blocks) > MAX_ARRAY_BLOCKS:
        raise ValueError("sidecar payload has too many native array blocks")
    block_metadata = []
    offset = 0
    raw_blocks = []
    for index, value in enumerate(blocks):
        descriptor = _array_descriptor(np, value)
        raw = value.tobytes(order="C")
        block_metadata.append(
            {
                "index": int(index),
                "dtype": descriptor["dtype"],
                "shape": descriptor["shape"],
                "order": "C",
                "offset": int(offset),
                "nbytes": int(len(raw)),
                "sha256": descriptor["sha256"],
            }
        )
        raw_blocks.append(raw)
        offset += len(raw)
    header = {
        "payload_schema_version": PAYLOAD_SCHEMA,
        "record": record_projection,
        "array_blocks": block_metadata,
        "logical_record_sha256": logical_record_sha256(np, record),
    }
    header_bytes = canonical_json_bytes(header)
    if len(header_bytes) > MAX_HEADER_BYTES:
        raise ValueError("sidecar payload header exceeds the fixed safety bound")
    payload = MAGIC + struct.pack(">I", len(header_bytes)) + header_bytes + b"".join(raw_blocks)
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("sidecar payload exceeds the fixed safety bound")
    return payload


def _require_exact_keys(mapping, expected, label):
    if not isinstance(mapping, dict) or set(mapping) != set(expected):
        observed = sorted(mapping) if isinstance(mapping, dict) else type(mapping).__name__
        raise ValueError("{} fields differ from the fixed payload schema: {}".format(label, observed))


def _strict_json_object(value):
    def no_duplicate_keys(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("sidecar payload JSON contains a duplicate key: {}".format(key))
            result[key] = item
        return result

    def reject_nonfinite(token):
        raise ValueError("sidecar payload JSON contains non-finite token: {}".format(token))

    try:
        parsed = json.loads(value.decode("utf-8"), object_pairs_hook=no_duplicate_keys, parse_constant=reject_nonfinite)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("sidecar payload header is not valid UTF-8 JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("sidecar payload header must be a JSON object")
    return parsed


def _validate_shape(shape, label):
    if not isinstance(shape, list) or len(shape) > 8:
        raise ValueError("{} shape is malformed".format(label))
    result = []
    for item in shape:
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise ValueError("{} shape contains an invalid dimension".format(label))
        result.append(int(item))
    return result


def _expected_nbytes(np, dtype_name, shape):
    count = 1
    for dimension in shape:
        count *= dimension
        if count > (1 << 62):
            raise ValueError("sidecar payload array shape overflows the safety bound")
    return int(count * np.dtype(_DTYPE_BY_NAME[dtype_name]).itemsize)


def _decode_blocks(np, raw, blocks):
    if not isinstance(blocks, list) or len(blocks) > MAX_ARRAY_BLOCKS:
        raise ValueError("sidecar payload block list is malformed")
    start = len(MAGIC) + HEADER_LENGTH_BYTES
    # The caller slices the raw data at the actual header boundary and passes
    # the complete payload plus that boundary as a separate value; see below.
    del start
    result = {}
    expected_offset = 0
    for expected_index, block in enumerate(blocks):
        _require_exact_keys(block, _BLOCK_KEYS, "array block {}".format(expected_index))
        if block["index"] != expected_index:
            raise ValueError("array block indices must be contiguous and ordered")
        dtype_name = block["dtype"]
        if dtype_name not in _DTYPE_BY_NAME or block["order"] != "C":
            raise ValueError("array block {} dtype/order is unsupported".format(expected_index))
        shape = _validate_shape(block["shape"], "array block {}".format(expected_index))
        nbytes = block["nbytes"]
        offset = block["offset"]
        if not isinstance(nbytes, int) or isinstance(nbytes, bool) or nbytes < 0:
            raise ValueError("array block nbytes is malformed")
        if not isinstance(offset, int) or isinstance(offset, bool) or offset != expected_offset:
            raise ValueError("array block offsets must be contiguous and ordered")
        if nbytes != _expected_nbytes(np, dtype_name, shape):
            raise ValueError("array block byte length disagrees with dtype/shape")
        end = offset + nbytes
        if end > len(raw):
            raise ValueError("array block extends beyond the payload")
        block_bytes = raw[offset:end]
        if not _is_sha256(block["sha256"]) or sha256_bytes(block_bytes) != block["sha256"]:
            raise ValueError("array block SHA-256 mismatch")
        wire_dtype = np.dtype(_DTYPE_BY_NAME[dtype_name])
        array = np.frombuffer(block_bytes, dtype=wire_dtype).reshape(tuple(shape), order="C")
        # Make an owned, native-endian array so downstream validation cannot
        # observe a read-only view into a malicious payload buffer.
        array = np.ascontiguousarray(array.astype(np.dtype(dtype_name), copy=True))
        result[expected_index] = array
        expected_offset = end
    if expected_offset != len(raw):
        raise ValueError("sidecar payload has trailing or unreferenced binary bytes")
    return result


def _rehydrate_arrays(value, arrays, consumed):
    if isinstance(value, dict):
        if "__array_block__" in value:
            _require_exact_keys(value, _PLACEHOLDER_KEYS, "array placeholder")
            index = value["__array_block__"]
            if not isinstance(index, int) or isinstance(index, bool) or index not in arrays or index in consumed:
                raise ValueError("array placeholder index is invalid or duplicated")
            array = arrays[index]
            descriptor = _array_descriptor_for_value(array)
            if (
                descriptor["dtype"] != value["dtype"]
                or descriptor["shape"] != value["shape"]
                or value["order"] != "C"
                or descriptor["sha256"] != value["sha256"]
            ):
                raise ValueError("array placeholder descriptor disagrees with binary block")
            consumed.add(index)
            return array
        return {str(key): _rehydrate_arrays(item, arrays, consumed) for key, item in value.items()}
    if isinstance(value, list):
        return [_rehydrate_arrays(item, arrays, consumed) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise ValueError("payload JSON contains an unsupported scalar type")


def _array_descriptor_for_value(value):
    # This helper is intentionally NumPy-free at definition time; callers have
    # already created an ndarray with a supported standard dtype.
    dtype_name = str(value.dtype)
    if dtype_name not in _DTYPE_BY_NAME or not value.flags.c_contiguous:
        raise ValueError("rehydrated array violates the fixed dtype/layout contract")
    return {
        "dtype": dtype_name,
        "shape": [int(item) for item in value.shape],
        "order": "C",
        "sha256": sha256_bytes(value.tobytes(order="C")),
    }


def decode_record(np, payload):
    """Safely decode and integrity-check a v2 payload into native arrays."""
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("sidecar payload must be bytes-like")
    payload = bytes(payload)
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("sidecar payload exceeds the fixed safety bound")
    prefix = len(MAGIC) + HEADER_LENGTH_BYTES
    if len(payload) < prefix or payload[: len(MAGIC)] != MAGIC:
        raise ValueError("sidecar payload magic/version mismatch")
    header_size = struct.unpack(">I", payload[len(MAGIC):prefix])[0]
    if header_size < 2 or header_size > MAX_HEADER_BYTES or prefix + header_size > len(payload):
        raise ValueError("sidecar payload header length is invalid")
    header_bytes = payload[prefix:prefix + header_size]
    header = _strict_json_object(header_bytes)
    if canonical_json_bytes(header) != header_bytes:
        raise ValueError("sidecar payload header is not canonical JSON")
    _require_exact_keys(
        header,
        ("payload_schema_version", "record", "array_blocks", "logical_record_sha256"),
        "sidecar payload header",
    )
    if header["payload_schema_version"] != PAYLOAD_SCHEMA or not _is_sha256(header["logical_record_sha256"]):
        raise ValueError("sidecar payload schema/hash declaration is invalid")
    if not isinstance(header["record"], dict):
        raise ValueError("sidecar payload record projection must be an object")
    arrays = _decode_blocks(np, payload[prefix + header_size:], header["array_blocks"])
    consumed = set()
    record = _rehydrate_arrays(header["record"], arrays, consumed)
    if consumed != set(arrays):
        raise ValueError("sidecar payload contains an unreferenced array block")
    if logical_record_sha256(np, record) != header["logical_record_sha256"]:
        raise ValueError("sidecar payload logical record hash mismatch")
    return record, header["logical_record_sha256"]
