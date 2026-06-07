from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

import sounddevice as sd

try:
    # 原有合成器：只负责左手手势和弦
    from .synth import MetronomeSynth, ThereminSynth as ChordSynth

    # 从 01e31... 搬入的合成器：负责右手旋律
    from .mixure_articulation_synth import ThereminSynth as MelodySynth
except ImportError:
    from synth import MetronomeSynth, ThereminSynth as ChordSynth
    from mixure_articulation_synth import ThereminSynth as MelodySynth


@dataclass
class AudioState:
    target_midi: int | None = None
    target_freq: float = 440.0
    target_volume: float = 0.0
    is_playing: bool = False

    articulation_id: int | None = None
    articulation_mode: str = "OFF"
    piano_sustain: bool = False

    pending_strong_clicks: int = 0
    pending_weak_clicks: int = 0

    pending_trigger_midis: list[int] | None = None
    pending_trigger_batches: list[tuple[list[int], float, float, float | None]] | None = None
    trigger_volume: float = 0.0
    trigger_seconds: float = 0.5
    trigger_release_seconds: float | None = None

    accompaniment_midis: list[int] | None = None
    accompaniment_volume: float = 0.0
    accompaniment_playing: bool = False


class AudioEngine:
    def __init__(self, sample_rate: int, block_size: int) -> None:
        self.sample_rate = sample_rate
        self.block_size = block_size

        # 右手旋律：01e31 钢琴 / 单簧管
        self.melody_synth = MelodySynth(sample_rate)

        # 左手手势和弦：39cb 原有合成器
        self.chord_synth = ChordSynth(sample_rate)

        self.metronome_synth = MetronomeSynth(sample_rate)
        self.state = AudioState()
        self._lock = threading.Lock()
        self.stream: sd.OutputStream | None = None
        self._recording_callback: Callable[[object, object, object], None] | None = None

    def _callback(self, outdata, frames, time_info, status) -> None:
        del time_info
        del status

        articulation_id = self.state.articulation_id
        articulation_mode = self.state.articulation_mode
        piano_sustain = self.state.piano_sustain

        with self._lock:
            target_midi = self.state.target_midi
            target_freq = self.state.target_freq
            target_volume = self.state.target_volume
            is_playing = self.state.is_playing

            articulation_id = self.state.articulation_id
            articulation_mode = self.state.articulation_mode
            piano_sustain = self.state.piano_sustain

            pending_strong_clicks = self.state.pending_strong_clicks
            pending_weak_clicks = self.state.pending_weak_clicks

            pending_trigger_midis = list(self.state.pending_trigger_midis or [])
            pending_trigger_batches = list(self.state.pending_trigger_batches or [])
            trigger_volume = self.state.trigger_volume
            trigger_seconds = self.state.trigger_seconds
            trigger_release_seconds = self.state.trigger_release_seconds

            accompaniment_midis = list(self.state.accompaniment_midis or [])
            accompaniment_volume = self.state.accompaniment_volume
            accompaniment_playing = self.state.accompaniment_playing

            self.state.pending_strong_clicks = 0
            self.state.pending_weak_clicks = 0
            self.state.pending_trigger_midis = []
            self.state.pending_trigger_batches = []

        # 右手旋律：使用 01e31 的钢琴单次触发 / 单簧管持续发声系统
        melody_block = self.melody_synth.render(
            frames,
            target_midi,
            target_freq,
            target_volume,
            is_playing,
            articulation_id,
            articulation_mode,
            piano_sustain,
        )

        # 左手手势和弦：继续使用 39cb 原有的触发音与持续和弦系统
        chord_block = self.chord_synth.render(
            frames,
            target_midi=None,
            target_freq=440.0,
            target_volume=0.0,
            is_playing=False,
            trigger_midis=pending_trigger_midis,
            trigger_volume=trigger_volume,
            trigger_seconds=trigger_seconds,
            trigger_release_seconds=trigger_release_seconds,
            trigger_batches=pending_trigger_batches,
            accompaniment_midis=accompaniment_midis,
            accompaniment_volume=accompaniment_volume,
            accompaniment_playing=accompaniment_playing,
        )

        metronome_block = self.metronome_synth.render(
            frames,
            pending_strong_clicks,
            pending_weak_clicks,
        )

        block = (melody_block + chord_block + metronome_block).clip(-1.0, 1.0)
        outdata[:, 0] = block
        recording_callback = self._recording_callback
        if recording_callback is not None:
            recording_callback(melody_block, chord_block, block)

       

    def start(self) -> None:
        if self.stream is not None:
            return

        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=1,
            dtype="float32",
            callback=self._callback,
            latency="low",
        )
        self.stream.start()

    def stop(self) -> None:
        if self.stream is None:
            return

        self.stream.stop()
        self.stream.close()
        self.stream = None

    def reset(self) -> None:
        with self._lock:
            self.state = AudioState()
            self.melody_synth.reset()
            self.chord_synth.reset()
            self.metronome_synth.reset()

    def update(
        self,
        target_midi: int | None,
        target_freq: float,
        target_volume: float,
        is_playing: bool,
        articulation_id: int | None = None,
        articulation_mode: str = "OFF",
        piano_sustain: bool = False,
        accompaniment_midis: list[int] | None = None,
        accompaniment_volume: float = 0.0,
        accompaniment_playing: bool = False,
    ) -> None:
        with self._lock:
            self.state.target_midi = None if target_midi is None else int(target_midi)
            self.state.target_freq = float(target_freq)
            self.state.target_volume = float(target_volume)
            self.state.is_playing = bool(is_playing)

            self.state.articulation_id = None if articulation_id is None else int(articulation_id)
            self.state.articulation_mode = str(articulation_mode or "OFF")
            self.state.piano_sustain = bool(piano_sustain)

            self.state.accompaniment_midis = [int(midi) for midi in (accompaniment_midis or [])]
            self.state.accompaniment_volume = float(accompaniment_volume)
            self.state.accompaniment_playing = bool(accompaniment_playing)

    def trigger_metronome(self, strong: bool) -> None:
        with self._lock:
            if strong:
                self.state.pending_strong_clicks += 1
            else:
                self.state.pending_weak_clicks += 1

    def trigger_notes(
        self,
        midis: list[int],
        volume: float,
        seconds: float,
        release_seconds: float | None = None,
    ) -> None:
        with self._lock:
            if self.state.pending_trigger_midis is None:
                self.state.pending_trigger_midis = []
            if self.state.pending_trigger_batches is None:
                self.state.pending_trigger_batches = []

            midi_batch = [int(midi) for midi in midis]
            self.state.pending_trigger_batches.append(
                (
                    midi_batch,
                    float(volume),
                    float(seconds),
                    None if release_seconds is None else float(release_seconds),
                )
            )
            self.state.trigger_volume = float(volume)
            self.state.trigger_seconds = float(seconds)
            self.state.trigger_release_seconds = None if release_seconds is None else float(release_seconds)

    def set_recording_callback(self, callback: Callable[[object, object, object], None] | None) -> None:
        self._recording_callback = callback
