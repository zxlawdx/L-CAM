"""Subprocess entrypoint: ONNX Runtime NEVER loads in the Vela process.

This module owns all native DirectML/CUDA sessions. It reads only binary
frames from stdin and writes framed results to stdout. Diagnostics go to
stderr, preventing accidental protocol corruption.
"""
from __future__ import annotations

import argparse
import logging
import sys

import numpy as np

from .inference import OnnxModelSpec, OnnxRuntimeEngine
from .onnx_wire import recv_packet, send_packet

log = logging.getLogger("lcam.onnx_child")


def serve(manifest: str, provider: str) -> int:
    writer = sys.stdout.buffer
    reader = sys.stdin.buffer
    try:
        spec = OnnxModelSpec.load(manifest)
        # This is the ONLY process allowed to initialize an ONNX session.
        engine = OnnxRuntimeEngine(spec, preference=provider)
        # The first actual GPU Run() can terminate Python natively on some
        # drivers. Execute a full single-tile warmup BEFORE reporting ready,
        # while Vela continues capturing frames with the classical filter.
        side = spec.tile_size or min(spec.input_width or 128, spec.input_height or 128)
        side = max(64, side)
        sample = np.full((side, side, 3), 80, dtype=np.uint8)
        output = engine.enhance(sample)
        if output.shape != sample.shape:
            raise RuntimeError("ONNX warmup retornou dimensões inesperadas.")
        send_packet(writer, {
            "ok": True, "type": "ready",
            "provider": engine.provider,
            "device_id": engine.dml_device_id if engine.provider == "DmlExecutionProvider" else None,
            "tile_size": spec.tile_size,
            "graph_profile": (
                "basic+fixed_dims"
                if engine.provider == "DmlExecutionProvider"
                and spec.sha256 == "231be201ab413dbc999d7951caa9844846b93a12a40a41e037d6b5888ed4e88c"
                else "default"
            ),
        })
    except Exception as exc:
        log.exception("Falha preparando ONNX no processo isolado")
        send_packet(writer, {"ok": False, "type": "error", "error": str(exc)[:1500]})
        return 2

    try:
        while True:
            packet = recv_packet(reader)
            if packet is None:
                return 0
            header, payload = packet
            if header.get("type") == "stop":
                return 0
            if header.get("type") != "frame":
                raise ValueError("Unknown ONNX subprocess command")
            height, width = header.get("height"), header.get("width")
            if (
                type(height) is not int or type(width) is not int
                or not 1 <= height <= 1080 or not 1 <= width <= 1920
                or len(payload) != height * width * 3
            ):
                raise ValueError("Invalid frame dimensions or payload")
            original = np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3).copy()
            processed = engine.enhance(original)
            if processed.shape != original.shape or processed.dtype != np.uint8:
                raise RuntimeError("Neural output has unexpected format")
            send_packet(writer, {
                "ok": True, "type": "frame", "height": height, "width": width,
            }, processed.tobytes(order="C"))
    except Exception as exc:
        # If native code terminates abruptly, the parent detects EOF and
        # switches to classic processing. Python errors also become safe.
        log.exception("ONNX child inference failed")
        try:
            send_packet(writer, {"ok": False, "type": "error", "error": str(exc)[:1500]})
        except (OSError, BrokenPipeError):
            pass
        return 3


def main() -> None:
    parser = argparse.ArgumentParser(description="Private L-CAM ONNX child process")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--provider", default="auto")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    sys.exit(serve(args.manifest, args.provider))


if __name__ == "__main__":
    main()
