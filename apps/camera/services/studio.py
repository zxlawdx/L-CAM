"""CameraStudio use cases: Vela-independent camera, preview and recording."""
from __future__ import annotations

import base64
import threading
from collections import deque
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import monotonic

from .capture import CameraCapture
from .contracts import CameraSource, FrameProcessor, PreviewEncoder, RecordingSink
from .encoding import JpegPreviewEncoder
from .filters import BasicEnhancer, ImageSettings
from .realtime import RealtimePipeline
from .model_manager import AsyncModelManager
from .recorder import VideoRecorder
from .recording_worker import RecordingWorker
from .audio import AudioCapture
from .ffmpeg_mux import AsyncFinalizer, FfmpegMuxer
from .presets import PresetStore


class CameraStudio:
    """Coordinates separate capture, processing and recording workers.

    Capture and processing never depend on how frequently JavaScript renders.
    Instantiating a different FrameProcessor or PreviewEncoder requires no
    changes in the Vela bridge or the application UI.
    """

    def __init__(
        self,
        camera: CameraSource | None = None,
        processor: FrameProcessor | None = None,
        recorder: RecordingSink | None = None,
        encoder: PreviewEncoder | None = None,
    ):
        self._camera = camera if camera is not None else CameraCapture()
        self._processor = processor if processor is not None else BasicEnhancer()
        self._recorder = recorder if recorder is not None else VideoRecorder()
        self._encoder = encoder if encoder is not None else JpegPreviewEncoder(86)
        self._recording = RecordingWorker(self._recorder)
        self._pipeline = None
        self._settings = ImageSettings()
        self._lock = threading.RLock()
        self._recording_path = None
        self._preview_times = deque(maxlen=45)
        self._models = AsyncModelManager(on_ready=self._install_processor)
        self._audio = None
        self._audio_path = None
        self._video_path = None
        self._finalizer = AsyncFinalizer()
        self._presets = PresetStore()

    def _install_processor(self, processor) -> None:
        with self._lock:
            self._processor = processor
            if self._pipeline is not None:
                self._pipeline.set_processor(processor)

    def _handle_processor_error(self, message: str) -> None:
        self._models.report_runtime_error(message)
        self._install_processor(BasicEnhancer())

    @staticmethod
    def _scunet_manifest() -> Path:
        return Path(__file__).resolve().parents[3] / "models" / "scunet_real_psnr" / "scunet_real_psnr.json"

    def model_catalog(self) -> dict:
        manifest = self._scunet_manifest()
        return {
            "models": [{
                "name": "SCUNet Real-PSNR",
                "description": "Redução de ruído real por blocos nativos — experimental",
                "installed": manifest.is_file(),
                "manifest": str(manifest) if manifest.is_file() else None,
            }]
        }

    def model_load_default(self, provider: str = "auto") -> dict:
        manifest = self._scunet_manifest()
        if not manifest.is_file():
            raise RuntimeError(
                "SCUNet ainda não instalado. Execute "
                "python -m scripts.install_scunet no ambiente do L-CAM."
            )
        return self._models.load(str(manifest), provider)

    def model_load(self, manifest_path: str, provider: str = "auto") -> dict:
        return self._models.load(manifest_path, provider)

    def model_unload(self) -> dict:
        return self._models.unload()

    def model_status(self) -> dict:
        return self._models.status()

    @staticmethod
    def audio_devices() -> list[dict]:
        return AudioCapture.list_devices()

    def presets_list(self) -> dict:
        return {"presets": self._presets.list()}

    def presets_save(self, name: str) -> dict:
        with self._lock:
            settings = self._settings
        return {"presets": self._presets.save(name, settings)}

    def presets_apply(self, name: str) -> dict:
        settings = self._presets.get(name)
        return self.configure(**asdict(settings))

    def presets_delete(self, name: str) -> dict:
        return {"presets": self._presets.delete(name)}

    def recording_status(self) -> dict:
        return {"writer": self._recording.stats(), "finalizer": self._finalizer.status()}

    @staticmethod
    def _fps(times) -> float:
        if len(times) < 2:
            return 0.0
        duration = times[-1] - times[0]
        return round((len(times) - 1) / duration, 1) if duration > 0 else 0.0

    def start(self, index: int = 0) -> dict:
        if type(index) is not int or not 0 <= index <= 10:
            raise ValueError("Índice de câmera inválido.")
        with self._lock:
            already_active = self._pipeline is not None
        if already_active:
            self.stop()
        with self._lock:
            self._camera.index = index
            pipeline = RealtimePipeline(
                camera=self._camera,
                processor=self._processor,
                recording_worker=self._recording,
                settings=self._settings,
                on_processor_error=self._handle_processor_error,
            )
            pipeline.start()
            self._pipeline = pipeline
            self._preview_times.clear()
            return {"active": True, "device": index, "workers": ["capture", "process"]}

    def configure(
        self, *, brightness=0, contrast=1.0, saturation=1.0,
        denoise=False, mode="off", strength=0.65, sharpness=0.0
    ) -> dict:
        # Explicit immutable validation before updating the active pipeline.
        settings = ImageSettings(
            brightness=brightness, contrast=contrast,
            saturation=saturation, denoise=denoise,
            mode=mode, strength=strength, sharpness=sharpness,
        )
        with self._lock:
            self._settings = settings
            if self._pipeline is not None:
                self._pipeline.configure(settings)
            return {"settings": asdict(settings)}

    def frame(self, original_preview: bool = False, last_sequence: int = -1) -> dict:
        if type(original_preview) is not bool:
            raise ValueError("original_preview deve ser booleano.")
        if type(last_sequence) is not int or last_sequence < -1:
            raise ValueError("last_sequence deve ser inteiro positivo ou -1.")
        with self._lock:
            pipeline = self._pipeline
        if pipeline is None:
            raise RuntimeError("Ative a câmera antes de solicitar quadros.")

        # Wait for new processed pixels without driving capture/processing.
        packet = pipeline.next_preview(last_sequence, timeout=0.25)
        if packet is None:
            state = pipeline.status()
            if state["error"] and not state["worker_alive"].get("lcam-process", False):
                raise RuntimeError(state["error"])
            latest = pipeline.latest()
            if latest is None:
                state = pipeline.status()
                if state["error"]:
                    raise RuntimeError(state["error"])
                return {"waiting": True, "sequence": last_sequence}
            if latest.sequence <= last_sequence:
                return {"unchanged": True, "sequence": last_sequence}
            packet = latest

        selected = packet.original if original_preview else packet.image
        jpeg = self._encoder.encode(selected)
        now = monotonic()
        with self._lock:
            self._preview_times.append(now)
            preview_fps = self._fps(self._preview_times)
        telemetry = pipeline.status()
        return {
            "image": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii"),
            "sequence": packet.sequence,
            "width": int(selected.shape[1]),
            "height": int(selected.shape[0]),
            "recording": self._recording.recording,
            "preview_fps": preview_fps,
            "capture_fps": telemetry["capture_fps"],
            "process_fps": telemetry["process_fps"],
            "processing_ms": round(packet.processing_ms, 1),
            "latency_ms": round((now - packet.captured_at) * 1000, 1),
            "dropped_frames": telemetry["dropped_preprocessing"],
            "mode": self._settings.mode,
        }

    def record_start(self, fps: float = 30.0, audio_device: int | None = None) -> dict:
        with self._lock:
            pipeline = self._pipeline
            if pipeline is None:
                raise RuntimeError("Ative a câmera antes de gravar.")
            if self._recording.recording:
                raise RuntimeError("Uma gravação já está ativa.")
            if self._finalizer.status()["state"] == "finalizing":
                raise RuntimeError("Aguarde a finalização da gravação anterior.")
        packet = pipeline.latest() or pipeline.next_preview(-1, timeout=1.0)
        if packet is None:
            raise RuntimeError("Aguarde o primeiro quadro antes de gravar.")
        if audio_device is not None and (type(audio_device) is not int or audio_device < 0):
            raise ValueError("Dispositivo de microfone inválido.")
        height, width = packet.image.shape[:2]
        folder = Path.home() / "Videos" / "L-CAM"
        filename = "lcam-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        final_path = folder / (filename + ".mp4")
        use_audio = audio_device is not None
        video_path = folder / (filename + ".video.mp4") if use_audio else final_path
        audio_path = folder / (filename + ".audio.wav") if use_audio else None
        mic = None
        if use_audio:
            if FfmpegMuxer().ffmpeg is None:
                raise RuntimeError("FFmpeg não encontrado. Instale FFmpeg para gravar áudio e vídeo.")
            mic = AudioCapture(device=audio_device)
            mic.start(str(audio_path))
        try:
            self._recording.start(str(video_path), float(fps), width, height)
        except Exception:
            if mic is not None:
                mic.stop()
            raise
        with self._lock:
            self._audio = mic
            self._audio_path = str(audio_path) if audio_path is not None else None
            self._video_path = str(video_path)
            self._recording_path = str(final_path)
        return {
            "recording": True, "path": str(final_path),
            "audio": use_audio, "audio_device": audio_device, "writer": "async",
        }

    def record_stop(self) -> dict:
        result = self._recording.stop()
        with self._lock:
            mic = self._audio
            audio_path = self._audio_path
            video_path = self._video_path
            path = self._recording_path
            self._audio = None
            self._audio_path = None
            self._video_path = None
            self._recording_path = None
        if mic is None:
            return {"recording": False, "path": path, "finalizing": False, **result}
        audio_stats = mic.stop()
        if not audio_stats["audio_started_at"] or not result["first_video_frame_at"]:
            raise RuntimeError(
                "Microfone ou vídeo sem quadros. Arquivos temporários preservados: "
                + str(video_path) + " ; " + str(audio_path)
            )
        self._finalizer.start(
            video_path, audio_path, path,
            video_started_at=result["first_video_frame_at"],
            audio_started_at=audio_stats["audio_started_at"],
        )
        return {
            "recording": False, "path": path, "finalizing": True,
            "audio": True, "audio_dropped_blocks": audio_stats["dropped_audio_blocks"],
            **result,
        }

    def stop(self) -> dict:
        with self._lock:
            pipeline = self._pipeline
            self._pipeline = None
        if pipeline is not None:
            pipeline.stop()
        if self._recording.recording or self._audio is not None:
            self.record_stop()
        with self._lock:
            self._recording_path = None
            self._preview_times.clear()
        return {"active": False}

    def status(self) -> dict:
        with self._lock:
            pipeline = self._pipeline
            settings = asdict(self._settings)
            path = self._recording_path
        return {
            "active": pipeline is not None,
            "recording": self._recording.recording,
            "settings": settings,
            "recording_path": path,
            "pipeline": pipeline.status() if pipeline is not None else None,
            "writer": self._recording.stats(),
            "model": self._models.status(),
            "finalizer": self._finalizer.status(),
        }

    def close(self) -> None:
        self._models.close()
        self.stop()
        self._finalizer.wait(timeout=12.0)
