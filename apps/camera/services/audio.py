"""Optional microphone capture with bounded callback-to-WAV queue.

sounddevice is only imported when the user explicitly enables audio.
Never perform disk writes in PortAudio's real-time callback.
"""
from __future__ import annotations

import importlib
import logging
import queue
import threading
import wave
from pathlib import Path
from time import monotonic

logger = logging.getLogger(__name__)


class AudioCapture:
    def __init__(self, device=None, sample_rate=48000, channels=1, capacity=256, sounddevice=None):
        if not isinstance(sample_rate, int) or not 8000 <= sample_rate <= 96000:
            raise ValueError("sample_rate inválido.")
        if channels not in (1, 2):
            raise ValueError("Somente mono ou estéreo são suportados.")
        self.device = device
        self.sample_rate = sample_rate
        self.channels = channels
        self._sd = sounddevice
        self._queue = queue.Queue(maxsize=capacity)
        self._stop = threading.Event()
        self._thread = None
        self._stream = None
        self._wave = None
        self.first_at = None
        self.dropped_blocks = 0
        self.errors = []
        self.path = None

    @staticmethod
    def list_devices(sounddevice=None) -> list[dict]:
        try:
            sd = sounddevice or importlib.import_module("sounddevice")
            return [
                {"id": i, "name": str(item["name"])[:120], "channels": int(item["max_input_channels"])}
                for i, item in enumerate(sd.query_devices())
                if int(item["max_input_channels"]) > 0
            ]
        except Exception as exc:
            logger.warning("Não foi possível enumerar microfones: %s", exc)
            return []

    def start(self, path: str) -> dict:
        if self._stream is not None:
            raise RuntimeError("Microfone já está capturando áudio.")
        try:
            sd = self._sd or importlib.import_module("sounddevice")
        except ImportError as exc:
            raise RuntimeError("Instale sounddevice para gravar com microfone: python -m pip install sounddevice") from exc
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._queue = queue.Queue(maxsize=self._queue.maxsize)
        self._stop.clear()
        self.first_at = None
        self.dropped_blocks = 0
        self.errors = []
        self.path = str(target)
        self._wave = wave.open(str(target), "wb")
        self._wave.setnchannels(self.channels)
        self._wave.setsampwidth(2)
        self._wave.setframerate(self.sample_rate)
        self._thread = threading.Thread(target=self._writer_loop, name="lcam-audio-write", daemon=True)
        self._thread.start()

        def callback(indata, frames, time_info, status):
            if self.first_at is None:
                self.first_at = monotonic()
            if status and len(self.errors) < 10:
                self.errors.append(str(status))
            try:
                self._queue.put_nowait(bytes(indata))
            except queue.Full:
                self.dropped_blocks += 1

        try:
            stream = sd.RawInputStream(
                device=self.device,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
                blocksize=1024,
                callback=callback,
            )
            stream.start()
            self._stream = stream
        except Exception:
            self._stop.set()
            self._thread.join(timeout=3)
            self._thread = None
            if self._wave is not None:
                self._wave.close()
                self._wave = None
            target.unlink(missing_ok=True)
            raise
        return {"active": True, "device": self.device, "sample_rate": self.sample_rate}

    def _writer_loop(self) -> None:
        while True:
            try:
                block = self._queue.get(timeout=0.1)
            except queue.Empty:
                if self._stop.is_set():
                    break
                continue
            try:
                self._wave.writeframesraw(block)
            except Exception as exc:
                logger.exception("Falha escrevendo WAV temporário.")
                self.errors.append(str(exc))
            finally:
                self._queue.task_done()

    def stop(self) -> dict:
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=8)
            if self._thread.is_alive():
                raise RuntimeError("Worker de áudio não foi finalizado.")
            self._thread = None
        if self._wave is not None:
            self._wave.close()
            self._wave = None
        return {
            "audio_path": self.path,
            "audio_started_at": self.first_at,
            "dropped_audio_blocks": self.dropped_blocks,
            "audio_errors": list(self.errors),
        }
