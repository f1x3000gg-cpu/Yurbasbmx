#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import wave
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import fluidsynth
import numpy as np
from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo
from sf2utils.sf2parse import Sf2File

SR = 44100
BPM = 160
BEAT = 60.0 / BPM
TPB = 480
BARS = 48
TAIL = 2.2
TOTAL_FRAMES = int((BARS * 4 * BEAT + TAIL) * SR)

PITCH = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

Event = Tuple[float, float, int, int]
Pattern = Sequence[Tuple[str | int | None, float]]


def mn(note: str | int) -> int:
    if isinstance(note, int):
        return note
    if len(note) >= 3 and note[1] in "#b":
        pitch, octave = note[:2], int(note[2:])
    else:
        pitch, octave = note[0], int(note[1:])
    return 12 * (octave + 1) + PITCH[pitch]


def hz(note: int) -> float:
    return 440.0 * 2.0 ** ((note - 69) / 12.0)


def add_note(events: Dict[str, List[Event]], track: str, start: float,
             note: str | int, duration: float, velocity: int) -> None:
    events[track].append((start, duration, mn(note), velocity))


def add_pattern(events: Dict[str, List[Event]], track: str, start: float,
                pattern: Pattern, velocity: int, shift: int = 0) -> None:
    cursor = start
    for note, duration in pattern:
        if note is not None:
            add_note(events, track, cursor, mn(note) + shift,
                     duration * 0.92, velocity)
        cursor += duration


def add_chord(events: Dict[str, List[Event]], track: str, start: float,
              notes: Sequence[str | int], duration: float, velocity: int) -> None:
    for note in notes:
        add_note(events, track, start, note, duration, velocity)


def preset_name(preset) -> str:
    value = getattr(preset, "name", "")
    if isinstance(value, bytes):
        return value.decode("latin1", errors="replace")
    return str(value)


def preset_bank(preset) -> int:
    return int(getattr(preset, "bank", 0))


def preset_program(preset) -> int:
    return int(getattr(preset, "preset", getattr(preset, "program", 0)))


def choose_preset(presets, keyword_sets: Sequence[Sequence[str]],
                  fallback_bank: int, fallback_program: int,
                  drum_only: bool = False):
    rows = []
    for preset in presets:
        name = preset_name(preset)
        normalized = name.casefold().replace("_", " ").replace("-", " ")
        rows.append((normalized, name, preset_bank(preset), preset_program(preset)))

    for keywords in keyword_sets:
        keys = [word.casefold() for word in keywords]
        matches = []
        for normalized, name, bank, program in rows:
            if drum_only and bank < 120:
                continue
            if all(key in normalized for key in keys):
                matches.append((len(normalized), name, bank, program))
        if matches:
            _, name, bank, program = sorted(matches)[0]
            return name, bank, program

    return f"fallback {fallback_bank}:{fallback_program}", fallback_bank, fallback_program


def build_events() -> Dict[str, List[Event]]:
    track_names = [
        "Romantic Trumpet", "Square Lead", "Piano 1", "Strings",
        "Choir Aahs", "French Horns", "Pipe Organ", "Orchestra Hit",
        "Timpani", "Synth Bass", "POWER Drums", "25% Pulse",
        "Triangle Bass",
    ]
    events: Dict[str, List[Event]] = {name: [] for name in track_names}

    # Original eight-bar hook. Its identity is rhythmic: a repeated off-beat
    # anchor, a leap, and a resolving answer. The notes are not generated.
    hook: List[Pattern] = [
        [(None, .25), ("A4", .25), ("A4", .5), ("C5", .5),
         ("D5", .25), ("C5", .25), ("G4", .5), ("Bb4", .5), ("A4", 1)],
        [(None, .25), ("A4", .25), ("A4", .5), ("F5", .5),
         ("E5", .25), ("D5", .25), ("C5", .5), ("Bb4", .5), ("G4", 1)],
        [(None, .25), ("G4", .25), ("G4", .5), ("Bb4", .5),
         ("C5", .25), ("Bb4", .25), ("F4", .5), ("A4", .5), ("G4", 1)],
        [("E4", .5), ("G4", .5), ("Bb4", .5), ("A4", .5),
         ("F4", .5), ("E4", .5), ("C#4", .5), ("D4", .5)],
        [("D5", .5), ("F5", .5), ("A5", 1), ("G5", .5),
         ("E5", .5), ("F5", 1)],
        [("C5", .5), ("E5", .5), ("G5", 1), ("F5", .5),
         ("D5", .5), ("E5", 1)],
        [("Bb4", .5), ("D5", .5), ("F5", .5), ("A5", .5),
         ("G5", .5), ("F5", .5), ("E5", 1)],
        [("C#5", .5), ("D5", .5), ("F5", 1), ("E5", .5),
         ("C#5", .5), ("D5", 1)],
    ]

    # Contrasting line for the middle. It has longer values and a different
    # contour, so the composition develops instead of repeating one riff.
    middle: List[Pattern] = [
        [("F5", 1), ("E5", .5), ("D5", .5), ("A4", 1), ("C5", 1)],
        [("Bb4", .5), ("D5", .5), ("F5", 1), ("E5", 1), ("C5", 1)],
        [("G4", .5), ("Bb4", .5), ("D5", 1), ("C5", .5),
         ("A4", .5), ("G4", 1)],
        [("A4", .5), ("C#5", .5), ("E5", 1), ("G5", .5),
         ("E5", .5), ("C#5", 1)],
        [("A5", 1), ("G5", .5), ("F5", .5), ("D5", 1), ("E5", 1)],
        [("F5", .5), ("E5", .5), ("C5", 1), ("Bb4", .5),
         ("A4", .5), ("F4", 1)],
        [("G4", .5), ("A4", .5), ("Bb4", 1), ("D5", .5),
         ("F5", .5), ("E5", 1)],
        [("C#5", .5), ("D5", .5), ("A4", 1), ("C#5", .5),
         ("E5", .5), ("D5", 1)],
    ]

    counter: List[Pattern] = [
        [("D4", 1), ("F4", 1), ("C4", 1), ("A3", 1)],
        [("C4", 1), ("F4", 1), ("E4", 1), ("C4", 1)],
        [("Bb3", 1), ("D4", 1), ("A3", 1), ("F3", 1)],
        [("A3", 1), ("C#4", 1), ("E4", 1), ("C#4", 1)],
        [("F4", 1), ("A4", 1), ("E4", 1), ("C4", 1)],
        [("E4", 1), ("G4", 1), ("D4", 1), ("Bb3", 1)],
        [("D4", 1), ("F4", 1), ("C4", 1), ("A3", 1)],
        [("A3", 1), ("C#4", 1), ("E4", 1), ("D4", 1)],
    ]

    chords = [
        ["D3", "A3", "C4", "F4"],
        ["Bb2", "F3", "A3", "D4"],
        ["G2", "D3", "F3", "Bb3"],
        ["A2", "E3", "G3", "C#4"],
        ["F2", "C3", "E3", "A3"],
        ["C3", "G3", "Bb3", "E4"],
        ["Bb2", "F3", "A3", "D4"],
        ["A2", "E3", "G3", "C#4"],
    ]
    roots = ["D2", "Bb1", "G1", "A1", "F1", "C2", "Bb1", "A1"]

    def add_bass_bar(bar: int, dense: bool) -> None:
        start = bar * 4
        idx = bar % 8
        root = roots[idx]
        if dense:
            rhythm = [0, .5, 1, 1.5, 2, 2.5, 3, 3.5]
            notes = [root, chords[idx][0], root, chords[idx][1],
                     root, chords[idx][2], chords[idx][1], root]
            for i, beat in enumerate(rhythm):
                add_note(events, "Synth Bass", start + beat, notes[i], .34,
                         94 if i in (0, 4) else 76)
        else:
            add_note(events, "Triangle Bass", start, root, 1.75, 92)
            add_note(events, "Triangle Bass", start + 2, root, 1.75, 80)

    def add_drums(bar: int, strength: float, fill: bool = False) -> None:
        start = bar * 4
        # Firm, readable backbeat. Aggression comes from accents and bass,
        # not from filling every possible subdivision.
        for beat, note, velocity in [
            (0, 36, 108), (1, 38, 116), (2, 36, 104), (3, 38, 120),
        ]:
            add_note(events, "POWER Drums", start + beat, note, .09,
                     min(127, int(velocity * strength)))
        for step in range(8):
            add_note(events, "POWER Drums", start + step * .5, 42, .05,
                     min(100, int((54 if step % 2 else 66) * strength)))
        if bar >= 8:
            add_note(events, "POWER Drums", start + 1.5, 36, .07,
                     min(110, int(78 * strength)))
            add_note(events, "POWER Drums", start + 3.5, 36, .07,
                     min(110, int(82 * strength)))
        if fill:
            for offset, drum_note, velocity in [
                (3.0, 45, 88), (3.25, 47, 94),
                (3.5, 48, 102), (3.75, 50, 112),
            ]:
                add_note(events, "POWER Drums", start + offset,
                         drum_note, .07, velocity)
        if bar in (0, 8, 16, 24, 36):
            add_note(events, "POWER Drums", start, 49, .20, 112)

    # 0-8: hook immediately, but with room around it.
    for bar in range(8):
        start = bar * 4
        idx = bar
        add_pattern(events, "Romantic Trumpet", start, hook[idx], 112)
        add_pattern(events, "Square Lead", start, hook[idx], 62, shift=-12)
        add_chord(events, "Piano 1", start, chords[idx], .34, 76)
        add_chord(events, "Strings", start, chords[idx], 3.72, 40)
        add_bass_bar(bar, False)
        add_drums(bar, .88, fill=(bar == 7))
        if bar in (0, 4):
            add_chord(events, "Orchestra Hit", start,
                      [chords[idx][0], chords[idx][2], chords[idx][3]], .20, 96)

    # 8-16: same hook with thicker Undertale-style orchestration.
    for bar in range(8, 16):
        start = bar * 4
        idx = bar - 8
        add_pattern(events, "Romantic Trumpet", start, hook[idx], 118)
        add_pattern(events, "Square Lead", start, counter[idx], 70, shift=12)
        add_chord(events, "Strings", start, chords[idx], 3.75, 54)
        if idx % 2 == 0:
            add_chord(events, "French Horns", start,
                      [chords[idx][0], chords[idx][2]], 1.45, 68)
        add_bass_bar(bar, True)
        add_drums(bar, 1.0, fill=(bar % 4 == 3))

    # 16-24: genuinely contrasting melodic section.
    for bar in range(16, 24):
        start = bar * 4
        idx = bar - 16
        add_pattern(events, "Romantic Trumpet", start, middle[idx], 112)
        add_pattern(events, "Piano 1", start, middle[idx], 62, shift=-12)
        add_pattern(events, "Square Lead", start, counter[idx], 56, shift=12)
        add_chord(events, "Strings", start, chords[idx], 3.75, 58)
        if idx in (0, 4):
            add_chord(events, "Choir Aahs", start, chords[idx], 3.72, 36)
        add_bass_bar(bar, True)
        add_drums(bar, 1.02, fill=(bar % 4 == 3))
        if idx in (3, 7):
            add_chord(events, "Orchestra Hit", start + 3.5,
                      [chords[idx][0], chords[idx][2], chords[idx][3]], .25, 108)

    # 24-32: return of hook with organ answering the end of each phrase.
    for bar in range(24, 32):
        start = bar * 4
        idx = bar - 24
        add_pattern(events, "Romantic Trumpet", start, hook[idx], 120)
        add_pattern(events, "Square Lead", start, hook[idx], 70, shift=-12)
        add_chord(events, "Strings", start, chords[idx], 3.75, 62)
        add_chord(events, "Choir Aahs", start, chords[idx], 3.75, 38)
        if idx % 2 == 1:
            add_chord(events, "Pipe Organ", start + 2,
                      [chords[idx][0], chords[idx][2]], 1.7, 46)
        add_bass_bar(bar, True)
        add_drums(bar, 1.08, fill=(bar % 4 == 3))

    # 32-36: short half-time pressure section, no sleepy piano solo.
    for bar in range(32, 36):
        start = bar * 4
        idx = bar - 32
        add_pattern(events, "Piano 1", start, middle[idx + 4], 96)
        add_chord(events, "Pipe Organ", start, chords[idx], 3.75, 58)
        add_chord(events, "Choir Aahs", start, chords[idx], 3.75, 48)
        add_note(events, "Timpani", start, mn(roots[idx]) + 12, .55, 96)
        add_note(events, "Timpani", start + 2, mn(roots[idx]) + 12, .45, 88)
        add_bass_bar(bar, False)
        # Half-time kick/snare, but still tense.
        add_note(events, "POWER Drums", start, 36, .10, 106)
        add_note(events, "POWER Drums", start + 2, 38, .12, 118)
        for step in range(8):
            add_note(events, "POWER Drums", start + step * .5, 42, .05,
                     48 if step % 2 else 60)

    # 36-44: final hook, strongest layer set, still readable.
    for bar in range(36, 44):
        start = bar * 4
        idx = bar - 36
        add_pattern(events, "Romantic Trumpet", start, hook[idx], 124)
        add_pattern(events, "Square Lead", start, hook[idx], 78, shift=-12)
        add_pattern(events, "Piano 1", start, counter[idx], 58, shift=12)
        add_chord(events, "Strings", start, chords[idx], 3.76, 66)
        add_chord(events, "Choir Aahs", start, chords[idx], 3.76, 44)
        if idx in (0, 4):
            add_chord(events, "French Horns", start,
                      [chords[idx][0], chords[idx][2]], 1.5, 80)
        add_bass_bar(bar, True)
        add_drums(bar, 1.12, fill=(bar % 2 == 1))
        if idx in (3, 7):
            add_chord(events, "Orchestra Hit", start + 3.5,
                      [chords[idx][0], chords[idx][2], chords[idx][3]], .30, 120)

    # 44-48: written ending tag, not another loop copy.
    ending: List[Pattern] = [
        [("A4", .5), ("C5", .5), ("D5", 1), ("F5", .5), ("E5", .5), ("D5", 1)],
        [("Bb4", .5), ("D5", .5), ("G5", 1), ("F5", .5), ("D5", .5), ("C5", 1)],
        [("A4", .5), ("C#5", .5), ("E5", 1), ("G5", .5), ("E5", .5), ("C#5", 1)],
        [("D5", .5), ("A4", .5), ("F4", .5), ("D4", .5), ("D5", 2)],
    ]
    for bar in range(44, 48):
        start = bar * 4
        idx = bar - 44
        chord_idx = (idx + 4) % 8
        add_pattern(events, "Romantic Trumpet", start, ending[idx], 122)
        add_pattern(events, "Square Lead", start, ending[idx], 70, shift=-12)
        add_chord(events, "Strings", start, chords[chord_idx], 3.75, 62)
        add_bass_bar(bar, True)
        add_drums(bar, 1.08, fill=(bar == 47))

    final_beat = 48 * 4 - .5
    add_chord(events, "Orchestra Hit", final_beat,
              ["D3", "A3", "C4", "F4", "D5"], .42, 127)
    add_chord(events, "Piano 1", final_beat,
              ["D3", "A3", "D4", "F4"], .48, 124)
    add_note(events, "Timpani", final_beat, "D3", .48, 127)
    add_note(events, "POWER Drums", final_beat, 49, .30, 127)

    return events


def build_preset_map(soundfont: Path):
    with soundfont.open("rb") as handle:
        presets = list(Sf2File(handle).presets)

    return {
        "Romantic Trumpet": choose_preset(
            presets,
            [["romantic", "tp"], ["romantic", "trumpet"],
             ["solo", "trumpet"], ["trumpet"]],
            0, 56,
        ),
        "Square Lead": choose_preset(
            presets,
            [["p5", "square"], ["mg", "square"], ["square", "wave"], ["square"]],
            0, 80,
        ),
        "Piano 1": choose_preset(presets, [["piano", "1"], ["piano"]], 0, 0),
        "Strings": choose_preset(presets, [["sgm", "strings"], ["strings"]], 0, 48),
        "Choir Aahs": choose_preset(presets, [["choir", "aahs"], ["choir"]], 0, 52),
        "French Horns": choose_preset(presets, [["french", "horn"], ["horn"]], 0, 60),
        "Pipe Organ": choose_preset(presets, [["pipe", "organ"], ["church", "organ"], ["organ"]], 0, 19),
        "Orchestra Hit": choose_preset(presets, [["orchestra", "hit"], ["orchestrahit"], ["impact", "hit"]], 0, 55),
        "Timpani": choose_preset(presets, [["timpani"]], 0, 47),
        "Synth Bass": choose_preset(presets, [["synth", "bass", "2"], ["synth", "bass"], ["fingered", "bass"]], 0, 39),
        "POWER Drums": choose_preset(presets, [["power"], ["standard", "1"], ["standard"]], 128, 16, drum_only=True),
    }


def render_sf(soundfont: Path, preset, events: Sequence[Event], gain: float) -> np.ndarray:
    synth = fluidsynth.Synth(gain=gain, samplerate=SR)
    sfid = synth.sfload(str(soundfont))
    _, bank, program = preset
    channel = 9 if bank >= 120 else 0
    selected = synth.program_select(channel, sfid, bank, program)
    if selected != 0:
        synth.program_select(channel, sfid, 128 if channel == 9 else 0, program)
    synth.set_reverb(roomsize=.14, damping=.70, width=.70, level=.08)
    synth.set_chorus(nr=2, level=.14, speed=.24, depth=1.8, type=0)

    timeline = []
    for start, duration, note, velocity in events:
        timeline.append((int(start * BEAT * SR), True, note, velocity))
        timeline.append((int((start + duration) * BEAT * SR), False, note, 0))
    timeline.sort(key=lambda row: (row[0], row[1]))

    chunks = []
    cursor = 0
    for frame, is_on, note, velocity in timeline:
        if frame > cursor:
            chunks.append(synth.get_samples(frame - cursor))
            cursor = frame
        if is_on:
            synth.noteon(channel, note, velocity)
        else:
            synth.noteoff(channel, note)
    if cursor < TOTAL_FRAMES:
        chunks.append(synth.get_samples(TOTAL_FRAMES - cursor))
    synth.delete()

    if not chunks:
        return np.zeros((TOTAL_FRAMES, 2), dtype=np.float64)
    return (np.concatenate(chunks).astype(np.float64) / 32768.0).reshape(-1, 2)


def render_25_pulse(events: Sequence[Event]) -> np.ndarray:
    audio = np.zeros((TOTAL_FRAMES, 2), dtype=np.float64)
    for start, duration, note, velocity in events:
        seconds = duration * BEAT
        count = int((seconds + .05) * SR)
        time = np.arange(count) / SR
        phase = (hz(note) * time) % 1.0
        signal = np.where(phase < .25, 1.0, -1.0)
        envelope = np.minimum(1.0, time / .002) * (.24 + .76 * np.exp(-time / .22))
        gate = min(count, int(seconds * SR))
        if gate < count:
            envelope[gate:] *= np.linspace(1.0, 0.0, count - gate)
        signal = np.round(signal * 12.0) / 12.0
        signal *= envelope * (velocity / 127.0) * .085
        begin = int(start * BEAT * SR)
        end = min(TOTAL_FRAMES, begin + count)
        audio[begin:end, 0] += signal[:end - begin] * .70
        audio[begin:end, 1] += signal[:end - begin] * .75
    return audio


def render_triangle(events: Sequence[Event]) -> np.ndarray:
    audio = np.zeros((TOTAL_FRAMES, 2), dtype=np.float64)
    for start, duration, note, velocity in events:
        seconds = duration * BEAT
        count = int((seconds + .04) * SR)
        time = np.arange(count) / SR
        phase = (hz(note) * time) % 1.0
        signal = 4.0 * np.abs(phase - .5) - 1.0
        signal = np.round(signal * 16.0) / 16.0
        envelope = np.minimum(1.0, time / .003) * (.20 + .80 * np.exp(-time / .34))
        gate = min(count, int(seconds * SR))
        if gate < count:
            envelope[gate:] *= np.linspace(1.0, 0.0, count - gate)
        signal *= envelope * (velocity / 127.0) * .15
        begin = int(start * BEAT * SR)
        end = min(TOTAL_FRAMES, begin + count)
        audio[begin:end, 0] += signal[:end - begin] * .72
        audio[begin:end, 1] += signal[:end - begin] * .70
    return audio


def save_wav(path: Path, audio: np.ndarray) -> None:
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak * .94
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes(pcm.tobytes())


def encode(wav_path: Path, mp3_path: Path, ogg_path: Path | None = None) -> None:
    filt = "highpass=f=28,lowpass=f=12500,acompressor=threshold=-17dB:ratio=2.3:attack=7:release=95,alimiter=limit=0.96"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path),
        "-af", filt, "-codec:a", "libmp3lame", "-q:a", "2", str(mp3_path),
    ], check=True)
    if ogg_path:
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path),
            "-af", filt, "-codec:a", "libvorbis", "-q:a", "6", str(ogg_path),
        ], check=True)


def export_midi(path: Path, events: Dict[str, List[Event]], presets) -> None:
    midi = MidiFile(ticks_per_beat=TPB)
    tempo = MidiTrack()
    tempo.append(MetaMessage("track_name", name="Tempo", time=0))
    tempo.append(MetaMessage("set_tempo", tempo=bpm2tempo(BPM), time=0))
    tempo.append(MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    midi.tracks.append(tempo)

    channels = {
        "Romantic Trumpet": 0, "Square Lead": 1, "Piano 1": 2,
        "Strings": 3, "Choir Aahs": 4, "French Horns": 5,
        "Pipe Organ": 6, "Orchestra Hit": 7, "Timpani": 8,
        "Synth Bass": 10, "POWER Drums": 9, "25% Pulse": 11,
        "Triangle Bass": 12,
    }
    for name, track_events in events.items():
        if not track_events:
            continue
        track = MidiTrack()
        track.append(MetaMessage("track_name", name=name, time=0))
        channel = channels[name]
        if name == "25% Pulse":
            bank, program = 0, 80
        elif name == "Triangle Bass":
            bank, program = 0, 38
        else:
            _, bank, program = presets[name]
        track.append(Message("control_change", channel=channel, control=0,
                             value=min(127, bank), time=0))
        track.append(Message("program_change", channel=channel,
                             program=min(127, program), time=0))
        timeline = []
        for start, duration, note, velocity in track_events:
            timeline.append((round(start * TPB), True, note, velocity))
            timeline.append((round((start + duration) * TPB), False, note, 0))
        timeline.sort(key=lambda row: (row[0], row[1]))
        previous = 0
        for tick, is_on, note, velocity in timeline:
            track.append(Message(
                "note_on" if is_on else "note_off",
                channel=channel, note=note,
                velocity=velocity if is_on else 0,
                time=tick - previous,
            ))
            previous = tick
        track.append(MetaMessage("end_of_track", time=1))
        midi.tracks.append(track)
    midi.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--soundfont", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    events = build_events()
    presets = build_preset_map(args.soundfont)

    preset_lines = ["Presets selected from UNDERTALE Soundfont 2:"]
    for label, (name, bank, program) in presets.items():
        preset_lines.append(f"{label}: {name} | bank={bank} preset={program}")
    (args.out / "SELECTED_UNDERTALE_PRESETS.txt").write_text(
        "\n".join(preset_lines) + "\n", encoding="utf-8"
    )

    full = np.zeros((TOTAL_FRAMES, 2), dtype=np.float64)
    proof = np.zeros_like(full)
    gains = {
        "Romantic Trumpet": .90, "Square Lead": .50, "Piano 1": .58,
        "Strings": .45, "Choir Aahs": .32, "French Horns": .42,
        "Pipe Organ": .38, "Orchestra Hit": .44, "Timpani": .50,
        "Synth Bass": .62, "POWER Drums": .70,
    }
    for label, gain in gains.items():
        stem = render_sf(args.soundfont, presets[label], events[label], .78)
        length = min(len(full), len(stem))
        full[:length] += stem[:length] * gain
        if label in ("Romantic Trumpet", "Square Lead", "Piano 1"):
            proof[:length] += stem[:length] * {
                "Romantic Trumpet": .96,
                "Square Lead": .42,
                "Piano 1": .55,
            }[label]

    full += render_25_pulse(events["25% Pulse"])
    full += render_triangle(events["Triangle Bass"])

    # Proof contains only the first eight bars of the written hook.
    proof_frames = int((8 * 4 * BEAT + 1.0) * SR)
    proof = proof[:proof_frames]

    full = np.tanh(full * 1.10)
    proof = np.tanh(proof * 1.04)

    full_wav = args.out / "KRIS_BONE_RIFF_FULL.wav"
    proof_wav = args.out / "KRIS_BONE_RIFF_MELODY_PROOF.wav"
    save_wav(full_wav, full)
    save_wav(proof_wav, proof)
    encode(full_wav, args.out / "KRIS_BONE_RIFF_FULL.mp3",
           args.out / "KRIS_BONE_RIFF_GAME.ogg")
    encode(proof_wav, args.out / "KRIS_BONE_RIFF_MELODY_PROOF.mp3")
    export_midi(args.out / "KRIS_BONE_RIFF.mid", events, presets)

    print("\n".join(preset_lines))
    print(f"Rendered at {BPM} BPM")


if __name__ == "__main__":
    main()
