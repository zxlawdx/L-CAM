"""Switchable live enhancement strategies without Vela/UI coupling."""
from __future__ import annotations

from .fast_live import FastLiveEnhancer
from .filters import BasicEnhancer


class ProfiledEnhancer:
    """A/B engine: fast default vs original heavy OpenCV/temporal pipeline.

    A single processor instance serves the realtime worker. Both internal
    engines are reused, and history is reset when the UI changes profiles.
    """

    def __init__(self):
        self.fast = FastLiveEnhancer()
        self.legacy = BasicEnhancer()
        self._active_profile = None

    def reset(self):
        self.fast.reset()
        self.legacy.reset()
        self._active_profile = None

    def process(self, frame, settings):
        profile = settings.profile
        if profile != self._active_profile:
            self.fast.reset()
            self.legacy.reset()
            self._active_profile = profile
        engine = self.fast if profile == "fast" else self.legacy
        return engine.process(frame, settings)
