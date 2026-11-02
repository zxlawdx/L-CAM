"""Audio input and mux are exercised with fake PortAudio/FFmpeg, no microphone."""
from __future__ import annotations

import json
import subprocess
import time
import wave
from pathlib import Path

import pytest

from apps.camera.services.audio import AudioCapture
from apps.camera.services.ffmpeg_mux import AsyncFinalizer, FfmpegMuxer


class FakePortAudio:
    @staticmethod
    def query_devices():
        return [
            {"name": "Webcam", "max_input_channels": 0},
            {"name": "Test Microphone", "max_input_channels": 1},
        ]

    class RawInputStream:
        def __init__(self, *, callback, **kwargs):
            self.callback = callback
            self.closed = False
        def start(self):
            self.callback(bytes(1024 * 2), 1024, None, "")
        def stop(self):
            pass
        def close(self):
            self.closed = True


def test_audio_writes_valid_wav_and_exposes_start_time(tmp_path):
    devices = AudioCapture.list_devices(sounddevice=FakePortAudio)
    assert len(devices) == 1 and devices[0]["id"] == 1
    audio = AudioCapture(device=1, sounddevice=FakePortAudio)
    target = tmp_path / "audio.wav"
    audio.start(str(target))
    summary = audio.stop()
    assert summary["audio_started_at"] is not None
    with wave.open(str(target), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getframerate() == 48000
        assert wav.getnframes() == 1024


def test_mux_offsets_audio_and_preserves_audio_video_tracks(tmp_path):
    video, audio, final = [tmp_path / name for name in ("source.mp4", "source.wav", "done.mp4")]
    video.write_bytes(b"video" * 40)
    audio.write_bytes(b"audio" * 40)
    commands = []

    def fake_runner(command, **kwargs):
        commands.append(command)
        if command[0] == "ffmpeg":
            Path(command[-1]).write_bytes(b"result" * 60)
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.CompletedProcess(command, 0, json.dumps({
            "streams": [{"codec_type": "video"}, {"codec_type": "audio"}]
        }), "")

    muxer = FfmpegMuxer(ffmpeg="ffmpeg", ffprobe="ffprobe", runner=fake_runner)
    assert muxer.mux(str(video), str(audio), str(final), video_started_at=10, audio_started_at=10.15) == str(final)
    assert final.exists()
    assert not video.exists() and not audio.exists()
    assert "-itsoffset" in commands[0]
    assert "libx264" in commands[0]


def test_mux_failure_preserves_original_files(tmp_path):
    video = tmp_path / "source.mp4"
    audio = tmp_path / "source.wav"
    video.write_bytes(b"video")
    audio.write_bytes(b"audio")
    def failed(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, "", "encoder not installed")
    muxer = FfmpegMuxer(ffmpeg="ffmpeg", ffprobe="ffprobe", runner=failed)
    with pytest.raises(RuntimeError, match="FFmpeg falhou"):
        muxer.mux(str(video), str(audio), str(tmp_path / "done.mp4"), video_started_at=10, audio_started_at=10)
    assert video.exists() and audio.exists()
