"""Safe asynchronous FFmpeg audio/video muxing.

Intermediate MP4+WAV files are preserved if mux/validation fails.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


class FfmpegMuxer:
    def __init__(self, ffmpeg=None, ffprobe=None, runner=None):
        self.ffmpeg = ffmpeg or shutil.which("ffmpeg")
        self.ffprobe = ffprobe or shutil.which("ffprobe")
        self._run = runner or subprocess.run

    def mux(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        *,
        video_started_at: float,
        audio_started_at: float,
    ) -> str:
        if not self.ffmpeg:
            raise RuntimeError("FFmpeg não encontrado no PATH; MP4 e WAV temporários foram preservados.")
        if video_started_at is None or audio_started_at is None:
            raise RuntimeError("Timestamps de início ausentes; arquivos temporários preservados.")
        video = Path(video_path)
        audio = Path(audio_path)
        final = Path(output_path)
        if not video.is_file() or not audio.is_file():
            raise RuntimeError("Arquivo temporário de vídeo ou áudio não encontrado.")
        temp = final.with_name(final.stem + ".muxing.mp4")
        delta = float(audio_started_at - video_started_at)
        cmd = [
            self.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-i", str(video),
        ]
        if delta > 0:
            cmd += ["-itsoffset", f"{delta:.6f}"]
        elif delta < 0:
            cmd += ["-ss", f"{-delta:.6f}"]
        cmd += [
            "-i", str(audio),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
            "-movflags", "+faststart", "-shortest", str(temp),
        ]
        result = self._run(cmd, capture_output=True, text=True, timeout=900, check=False)
        if result.returncode != 0 or not temp.is_file() or temp.stat().st_size < 100:
            raise RuntimeError("FFmpeg falhou: " + (result.stderr or "arquivo de saída inválido")[-600:])
        if self.ffprobe:
            probe = self._run(
                [self.ffprobe, "-v", "error", "-show_entries",
                 "stream=codec_type", "-of", "json", str(temp)],
                capture_output=True, text=True, timeout=30, check=False,
            )
            if probe.returncode != 0:
                raise RuntimeError("ffprobe não conseguiu verificar o MP4.")
            try:
                streams = json.loads(probe.stdout).get("streams", [])
                media_types = {item["codec_type"] for item in streams}
            except (ValueError, KeyError, TypeError) as exc:
                raise RuntimeError("ffprobe retornou metadados inválidos.") from exc
            if not {"video", "audio"}.issubset(media_types):
                raise RuntimeError("O MP4 gerado não contém streams válidos de vídeo e áudio.")
        os.replace(temp, final)
        # Clean-up only after validation and atomic move succeeded.
        video.unlink(missing_ok=True)
        audio.unlink(missing_ok=True)
        return str(final)


class AsyncFinalizer:
    """Nonblocking finalization and queryable state for Vela UI."""

    def __init__(self, muxer=None):
        self._muxer = muxer or FfmpegMuxer()
        self._lock = threading.RLock()
        self._thread = None
        self._state = {"state": "idle", "output": None, "error": None}

    def start(self, video, audio, output, *, video_started_at, audio_started_at) -> dict:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("Outra gravação ainda está sendo finalizada.")
            self._state = {"state": "finalizing", "output": str(output), "error": None}
        def task():
            try:
                path = self._muxer.mux(
                    video, audio, output,
                    video_started_at=video_started_at, audio_started_at=audio_started_at,
                )
                with self._lock:
                    self._state = {"state": "done", "output": path, "error": None}
            except Exception as exc:
                logger.exception("Falha no mux de áudio e vídeo; originais preservados.")
                with self._lock:
                    self._state = {
                        "state": "error", "output": str(output), "error": str(exc)[:800],
                        "video_backup": str(video), "audio_backup": str(audio),
                    }
        self._thread = threading.Thread(target=task, name="lcam-ffmpeg-mux", daemon=True)
        self._thread.start()
        return self.status()

    def status(self) -> dict:
        with self._lock:
            return dict(self._state)
