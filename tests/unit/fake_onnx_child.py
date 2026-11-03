"""Test-only IPC worker. Runs without ONNX Runtime and has crash modes."""
from __future__ import annotations

import argparse
import os
import sys
import time

from apps.camera.services.onnx_wire import recv_packet, send_packet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=(
        "echo", "crash-init", "crash-frame", "hang", "error", "bad-size"
    ), default="echo")
    mode = parser.parse_args().mode
    if mode == "crash-init":
        os._exit(29)
    send_packet(sys.stdout.buffer, {
        "type": "ready", "ok": True, "provider": "CPUExecutionProvider",
        "device_id": None,
    })
    while True:
        packet = recv_packet(sys.stdin.buffer)
        if packet is None:
            return
        header, payload = packet
        if header.get("type") == "stop":
            return
        if mode == "crash-frame":
            os._exit(23)
        if mode == "hang":
            time.sleep(8)
        if mode == "error":
            send_packet(sys.stdout.buffer, {
                "type": "error", "ok": False, "error": "Fake ONNX failure",
            })
            return
        if mode == "bad-size":
            payload = payload[:-1]
        send_packet(sys.stdout.buffer, {
            "type": "frame", "ok": True,
            "height": header["height"], "width": header["width"],
        }, payload)


if __name__ == "__main__":
    main()
