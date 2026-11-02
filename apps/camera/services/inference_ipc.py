"""Length-prefixed JSON plus raw BGR frames for the isolated ONNX worker.

No pickle or arbitrary object deserialization. Explicit packet size limits.
"""
from __future__ import annotations

import json
import struct

MAX_HEADER = 64 * 1024
MAX_IMAGE = 16 * 1024 * 1024


def read_exact(stream, size: int) -> bytes:
    pieces = []
    while size:
        part = stream.read(size)
        if not part:
            raise EOFError("Worker ONNX fechou o canal de comunicação.")
        pieces.append(part)
        size -= len(part)
    return b"".join(pieces)


def write_packet(stream, message: dict, pixels: bytes = b"") -> None:
    header = json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if not 0 < len(header) <= MAX_HEADER or len(pixels) > MAX_IMAGE:
        raise ValueError("Pacote ONNX excede os limites do protocolo.")
    stream.write(struct.pack("<II", len(header), len(pixels)))
    stream.write(header)
    if pixels:
        stream.write(pixels)
    stream.flush()


def read_packet(stream) -> tuple[dict, bytes]:
    length_header, length_pixels = struct.unpack("<II", read_exact(stream, 8))
    if not 0 < length_header <= MAX_HEADER or length_pixels > MAX_IMAGE:
        raise ValueError("Cabeçalho ONNX inválido ou excessivamente grande.")
    message = json.loads(read_exact(stream, length_header).decode("utf-8"))
    if not isinstance(message, dict):
        raise ValueError("Resposta ONNX inválida (JSON não é objeto).")
    return message, read_exact(stream, length_pixels)
