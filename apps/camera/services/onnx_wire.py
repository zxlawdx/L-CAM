"""Bounded binary protocol for an isolated ONNX process.

Framing: uint32 big-endian JSON header size, JSON header, uint32 binary
payload size, bytes. No pickle, no executable commands or arbitrary paths.
"""
from __future__ import annotations

import json
import struct

MAX_HEADER = 64 * 1024
MAX_FRAME = 24 * 1024 * 1024


def read_exact(stream, size: int, *, allow_eof: bool = False):
    if size < 0:
        raise ValueError("Negative packet size")
    parts = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            if allow_eof and remaining == size:
                return None
            raise EOFError("Worker pipe closed during packet transfer")
        parts.append(chunk)
        remaining -= len(chunk)
    return b"".join(parts)


def recv_packet(stream):
    prefix = read_exact(stream, 4, allow_eof=True)
    if prefix is None:
        return None
    header_len = struct.unpack(">I", prefix)[0]
    if not 2 <= header_len <= MAX_HEADER:
        raise ValueError("Invalid ONNX packet header size")
    header = json.loads(read_exact(stream, header_len).decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("ONNX packet header must be an object")
    payload_len = struct.unpack(">I", read_exact(stream, 4))[0]
    if payload_len > MAX_FRAME:
        raise ValueError("ONNX frame exceeds configured limit")
    return header, read_exact(stream, payload_len) if payload_len else b""


def send_packet(stream, header: dict, payload: bytes = b""):
    blob = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if not 2 <= len(blob) <= MAX_HEADER or len(payload) > MAX_FRAME:
        raise ValueError("ONNX packet exceeds configured limit")
    stream.write(struct.pack(">I", len(blob)))
    stream.write(blob)
    stream.write(struct.pack(">I", len(payload)))
    if payload:
        stream.write(payload)
    stream.flush()
