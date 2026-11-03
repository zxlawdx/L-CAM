"""Crash-isolated ONNX engine, protecting Vela/pywebview from native GPU faults.

A native DirectML access violation cannot be handled with Python try/except
inside the same process. ONNX Runtime therefore lives in a long-lived child
process; the Vela process only exchanges BGR frames via bounded IPC.
"""
from __future__ import annotations

import logging
import queue
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path

import numpy as np

from .onnx_wire import recv_packet, send_packet

log = logging.getLogger(__name__)


def _native_exit(returncode):
    if returncode is None:
        return "processo encerrado ou sem resposta"
    code = returncode & 0xFFFFFFFF
    known = {
        0xC0000005: "violação de acesso à memória (0xC0000005)",
        0xC0000017: "memória insuficiente (0xC0000017)",
        0xC0000409: "abortamento do processo nativo (0xC0000409)",
        0xC0000135: "DLL nativa ausente (0xC0000135)",
        0xC0000142: "falha inicializando DLL (0xC0000142)",
    }
    return known.get(code, f"código de saída {returncode} (0x{code:08X})")


class IsolatedOnnxRuntimeEngine:
    """Drop-in adapter compatible with NeuralEnhancementProcessor.

    process() is called by the existing one-consumer pipeline worker.
    The bounded IPC and timeouts mean one expensive frame cannot cause an
    ever-growing queue of stale video. On timeout/crash, the caller triggers
    the BasicEnhancer fallback.
    """

    def __init__(
        self, spec, preference="auto", *,
        command=None, startup_timeout=65, frame_timeout=8,
        process_factory=subprocess.Popen,
    ):
        spec.validate()
        if not 1 <= startup_timeout <= 300 or not 1 <= frame_timeout <= 180:
            raise ValueError("Tempo limite do processo ONNX inválido.")
        self.spec = spec
        self.provider = None
        self.dml_device_id = None
        self._process = None
        self._closed = False
        self._messages = queue.Queue(maxsize=2)
        self._stderr_tail = deque(maxlen=30)
        self._lock = threading.RLock()
        self._close_lock = threading.Lock()
        self._frame_timeout = frame_timeout
        if command is None:
            if not isinstance(getattr(spec, "manifest_path", None), Path):
                raise ValueError("ONNX isolado exige o manifesto verificado do modelo.")
            command = [
                sys.executable, "-u", "-m", "apps.camera.services.onnx_child",
                "--manifest", str(spec.manifest_path),
                "--provider", preference,
            ]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        try:
            self._process = process_factory(
                command,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, bufsize=0,
                creationflags=creationflags,
                cwd=str(Path(__file__).resolve().parents[3]),
            )
            self._reader = threading.Thread(
                target=self._read_loop, name="lcam-onnx-rx", daemon=True,
            )
            self._reader.start()
            self._errors = threading.Thread(
                target=self._stderr_loop, name="lcam-onnx-stderr", daemon=True,
            )
            self._errors.start()
            msg, payload = self._wait(startup_timeout)
            if not msg.get("ok") or msg.get("type") != "ready":
                raise RuntimeError(msg.get("error") or "Worker ONNX não inicializou.")
            self.provider = str(msg["provider"])
            self.dml_device_id = msg.get("device_id")
            log.info("ONNX isolado iniciado PID=%s, provider=%s", self._process.pid, self.provider)
        except Exception:
            self.close()
            raise

    def _read_loop(self):
        try:
            while True:
                packet = recv_packet(self._process.stdout)
                if packet is None:
                    break
                self._messages.put(packet)
        except Exception as exc:
            self._messages.put(({"ok": False, "error": "Falha de IPC ONNX: " + str(exc)}, b""))
        finally:
            try:
                self._messages.put_nowait(({"ok": False, "error": "Processo ONNX encerrou."}, b""))
            except queue.Full:
                pass

    def _stderr_loop(self):
        try:
            for raw in iter(self._process.stderr.readline, b""):
                message = raw.decode("utf-8", errors="replace").rstrip()
                if message:
                    self._stderr_tail.append(message[-1200:])
                    log.warning("[ONNX subprocess] %s", message[:900])
        except (OSError, ValueError):
            pass

    def _failure(self, cause: str) -> RuntimeError:
        code = self._process.poll() if self._process else None
        tail = " | ".join(list(self._stderr_tail)[-3:])[:900]
        return RuntimeError(
            f"{cause}. Subprocesso ONNX: {_native_exit(code)}."
            + (f" Detalhes: {tail}" if tail else "")
            + " Vela continua no processamento clássico."
        )

    def _wait(self, timeout):
        try:
            return self._messages.get(timeout=timeout)
        except queue.Empty:
            raise self._failure(f"Timeout ONNX após {timeout}s") from None

    def enhance(self, frame):
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("Quadro para ONNX deve ser BGR uint8.")
        h, w = frame.shape[:2]
        if not 1 <= h <= 1080 or not 1 <= w <= 1920:
            raise ValueError("Quadro excede o limite de inferência ONNX.")
        with self._lock:
            if self._closed or self._process is None or self._process.poll() is not None:
                raise self._failure("Processo ONNX indisponível")
            try:
                send_packet(self._process.stdin, {
                    "type": "frame", "height": h, "width": w,
                }, frame.tobytes(order="C"))
                response, output = self._wait(self._frame_timeout)
                if not response.get("ok"):
                    raise self._failure(response.get("error", "Falha no worker ONNX"))
                if (
                    response.get("type") != "frame"
                    or response.get("height") != h
                    or response.get("width") != w
                    or len(output) != h * w * 3
                ):
                    raise self._failure("Formato do quadro ONNX retornado é inválido")
                return np.frombuffer(output, dtype=np.uint8).reshape(h, w, 3).copy()
            except (RuntimeError, OSError, EOFError, BrokenPipeError) as exc:
                error = self._failure(str(exc))
                self.close()
                raise error from exc

    def close(self):
        # Do NOT acquire the frame-operation lock here: a native GPU call may
        # be stuck. Closing from the Vela UI must terminate the child promptly
        # without waiting for the inference timeout or for video frame locks.
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            proc = self._process
            if proc is None:
                return
            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                log.exception("Não foi possível finalizar processo ONNX PID=%s", proc.pid)
            finally:
                for stream in (proc.stdin, proc.stdout, proc.stderr):
                    if stream:
                        try:
                            stream.close()
                        except (OSError, ValueError):
                            pass
