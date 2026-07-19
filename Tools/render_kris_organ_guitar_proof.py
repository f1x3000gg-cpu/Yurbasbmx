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

SR = 44100
BPM = 148
BEAT_SECONDS = 60.0 / BPM
BARS = 8
TAIL_SECONDS = 2.2
TPB = 480

PITCH = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

Event = Tuple[float, float, int, int]
Pattern = Sequence[Tuple[str | int | None, float]]

PRESETS = {
    "Megalo Over Guitar": (0, 0, "100 MEGALOVANIA - Overdriven Guitar"),
    "Megalo Square25": (0, 8, "100 MEGALOVANIA - 25% Square"),
    "Megalo Impact": (0, 9, "100 MEGALOVANIA - Impact Hit"),
    "Megalo Background Guitar": (0, 13, "100 MEGALOVANIA - Background Guitar"),
    "Megalo Bass": (0, 14, "100 MEGALOVANIA - Bass"),
    "Hopes Piano1": (2, 126, "087 Hopes and Dreams - Piano 1"),
    "Hopes POWER": (2, 127, "087 Hopes and Dreams - POWER DrumKit"),
    "Hopes Pulse25": (3, 5, "087 Hopes and Dreams - Pulse 25%"),
    "Hopes Duty": (3, 7, "087 Hopes and Dreams - Duty Cycle"),
    "Hopes Lead Guitar": (3, 8, "087 Hopes and Dreams - Lead Guitar"),
}


def note_number(note: str | int) -> int:
    if isinstance(note, int):
        return note
    if len(note) >= 3 and note[1] in "#b":
        name, octave = note[:2], int(note[2:])
    else:
        name, octave = note[0], int(note[1:])
    return 12 * (octave + 1) + PITCH[name]


def add_note(events: Dict[str, List[Event]], track: str, start: float,
             note: str | int, duration: float, velocity: int) -> None:
    events[track].append((float(start), float(duration), note_number(note), int(velocity)))


def add_pattern(events: Dict[str, List[Event]], track: str, start: float,
                pattern: Pattern, velocity: int, transpose: int = 0) -> None:
    cursor = float(start)
    for note, duration in pattern:
        if note is not None:
            add_note(events, track, cursor, note_number(note) + transpose,
                     duration * 0.92, velocity)
        cursor += duration


def add_chord(events: Dict[str, List[Event]], track: str, start: float,
              notes: Sequence[str], duration: float, velocity: int) -> None:
    for note in notes:
        add_note(events, track, start, note, duration, velocity)


def build_events() -> Dict[str, List[Event]]:
    events = {name: [] for name in PRESETS}

    melody: List[Pattern] = [
        [(None, .5), ("B4", .75), ("E5", .25), ("G5", 1.0), ("F#5", .5), ("E5", 1.0)],
        [("D5", .75), ("B4", .25), ("A4", .5), ("B4", 1.0), (None, .5), ("E5", 1.0)],
        [("G4", .5), ("B4", 1.0), ("D5", .5), ("E5", .75), ("D5", .25), ("B4", 1.0)],
        [("A4", 1.0), ("C5", .5), ("B4", .5), ("G4", .75), ("F#4", .25), ("E4", 1.0)],
        [("E5", .5), ("G5", .5), ("B5", 1.0), ("A5", .75), ("G5", .25), ("F#5", 1.0)],
        [("D5", .5), ("F#5", .5), ("A5", 1.0), ("G5", .5), ("E5", .5), ("B4", 1.0)],
        [("C5", .75), ("E5", .25), ("G5", .5), ("F#5", 1.0), ("D#5", .5), ("E5", 1.0)],
        [("B4", .5), ("D5", .5), ("E5", 1.0), ("G5", .5), ("F#5", .5), ("E5", 1.0)],
    ]

    chords = [
        ["E3", "B3", "D4", "G4"],
        ["C3", "G3", "B3", "E4"],
        ["G2", "D3", "F#3", "B3"],
        ["D3", "A3", "C4", "F#4"],
        ["E3", "B3", "D4", "G4"],
        ["B2", "F#3", "A3", "D4"],
        ["C3", "G3", "B3", "E4"],
        ["B2", "F#3", "A3", "D#4"],
    ]
    roots = ["E2", "C2", "G1", "D2", "E2", "B1", "C2", "B1"]

    for bar in range(BARS):
        start = bar * 4.0
        add_pattern(events, "Hopes Piano1", start, melody[bar], 104)
        add_pattern(events, "Hopes Lead Guitar", start, melody[bar], 72)
        if bar in (0, 2, 4, 6):
            add_pattern(events, "Hopes Pulse25", start, melody[bar], 24, transpose=-12)
        else:
            add_pattern(events, "Megalo Square25", start, melody[bar], 24, transpose=-12)

        add_chord(events, "Hopes Piano1", start, chords[bar], .38, 42)
        add_chord(events, "Hopes Piano1", start + 2.5, chords[bar], .34, 36)

        root = note_number(roots[bar])
        bass_pattern = [
            (0.0, root), (1.0, root + 7), (1.5, root + 12),
            (2.5, root + 7), (3.25, root + (10 if bar in (3, 7) else 12)),
        ]
        for pos, pitch in bass_pattern:
            add_note(events, "Megalo Bass", start + pos, pitch, .48, 92 if pos == 0 else 72)

        guitar_root = root + 12
        power = [guitar_root, guitar_root + 7, guitar_root + 12]
        for pos in (0.0, .75, 2.0, 3.0):
            for pitch in power:
                add_note(events, "Megalo Background Guitar", start + pos, pitch, .24,
                         69 if pos in (0.0, 2.0) else 52)
        if bar >= 4:
            add_pattern(events, "Megalo Over Guitar", start, melody[bar], 52, transpose=-12)

        for pos in (0.0, 2.0, 2.75):
            add_note(events, "Hopes POWER", start + pos, 36, .08, 104 if pos != 2.75 else 72)
        for pos in (1.0, 3.0):
            add_note(events, "Hopes POWER", start + pos, 38, .09, 114)
        for step in range(8):
            add_note(events, "Hopes POWER", start + step * .5, 42, .045,
                     44 if step % 2 else 58)
        if bar == 7:
            for off, drum in ((3.0, 45), (3.25, 47), (3.5, 48), (3.75, 50)):
                add_note(events, "Hopes POWER", start + off, drum, .06, 82)
        if bar in (0, 4, 7):
            add_chord(events, "Megalo Impact", start,
                      [chords[bar][0], chords[bar][2]], .18, 86)

    return events


def render_stem(soundfont: Path, bank: int, preset: int,
                events: Sequence[Event], gain: float = .78) -> np.ndarray:
    total_frames = int(((BARS * 4 * BEAT_SECONDS) + TAIL_SECONDS) * SR)
    synth = fluidsynth.Synth(gain=gain, samplerate=SR)
    sfid = synth.sfload(str(soundfont))
    result = synth.program_select(0, sfid, bank, preset)
    if result != 0:
        synth.delete()
        raise RuntimeError(f"Cannot select bank={bank}, preset={preset}")
    synth.set_reverb(roomsize=.12, damping=.70, width=.66, level=.065)
    synth.set_chorus(nr=2, level=.10, speed=.24, depth=1.6, type=0)

    timeline = []
    for start, duration, note, velocity in events:
        timeline.append((int(start * BEAT_SECONDS * SR), 1, note, velocity))
        timeline.append((int((start + duration) * BEAT_SECONDS * SR), 0, note, 0))
    timeline.sort(key=lambda item: (item[0], item[1]))

    chunks = []
    cursor = 0
    for frame, is_on, note, velocity in timeline:
        if frame > cursor:
            chunks.append(synth.get_samples(frame - cursor))
            cursor = frame
        if is_on:
            synth.noteon(0, note, velocity)
        else:
            synth.noteoff(0, note)
    if cursor < total_frames:
        chunks.append(synth.get_samples(total_frames - cursor))
    synth.delete()

    if not chunks:
        return np.zeros((total_frames, 2), dtype=np.float64)
    raw = np.concatenate(chunks).astype(np.float64) / 32768.0
    return raw.reshape(-1, 2)


def write_wav(path: Path, audio: np.ndarray) -> None:
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak * .94
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes(pcm.tobytes())


def encode_mp3(wav_path: Path, mp3_path: Path) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path),
        "-af", "highpass=f=30,lowpass=f=12500,acompressor=threshold=-18dB:ratio=2:attack=9:release=120,alimiter=limit=0.96",
        "-codec:a", "libmp3lame", "-q:a", "2", str(mp3_path),
    ], check=True)


def export_midi(path: Path, events: Dict[str, List[Event]]) -> None:
    midi = MidiFile(ticks_per_beat=TPB)
    tempo = MidiTrack()
    tempo.append(MetaMessage("track_name", name="Tempo", time=0))
    tempo.append(MetaMessage("set_tempo", tempo=bpm2tempo(BPM), time=0))
    tempo.append(MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    midi.tracks.append(tempo)

    channel = 0
    for name, track_events in events.items():
        if not track_events:
            continue
        while channel == 9:
            channel += 1
        midi_channel = channel % 16
        channel += 1
        track = MidiTrack()
        track.append(MetaMessage("track_name", name=name, time=0))
        bank, preset, _ = PRESETS[name]
        track.append(Message("control_change", channel=midi_channel,
                             control=0, value=min(127, bank), time=0))
        track.append(Message("program_change", channel=midi_channel,
                             program=min(127, preset), time=0))
        timeline = []
        for start, duration, note, velocity in track_events:
            timeline.append((round(start * TPB), 1, note, velocity))
            timeline.append((round((start + duration) * TPB), 0, note, 0))
        timeline.sort(key=lambda item: (item[0], item[1]))
        previous = 0
        for tick, is_on, note, velocity in timeline:
            track.append(Message("note_on" if is_on else "note_off",
                                 channel=midi_channel, note=note,
                                 velocity=velocity if is_on else 0,
                                 time=tick - previous))
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
    total = int(((BARS * 4 * BEAT_SECONDS) + TAIL_SECONDS) * SR)
    context = np.zeros((total, 2), dtype=np.float64)
    melody = np.zeros_like(context)

    context_gains = {
        "Megalo Over Guitar": .48,
        "Megalo Square25": .22,
        "Megalo Impact": .42,
        "Megalo Background Guitar": .42,
        "Megalo Bass": .64,
        "Hopes Piano1": .66,
        "Hopes POWER": .68,
        "Hopes Pulse25": .20,
        "Hopes Duty": .0,
        "Hopes Lead Guitar": .58,
    }
    melody_gains = {
        "Hopes Piano1": .86,
        "Hopes Lead Guitar": .22,
    }

    selected = [
        f"BPM: {BPM}",
        "Key: E minor",
        "Completely new melody; not a variation of the previous D-minor proof.",
        "Only MEGALOVANIA and Hopes and Dreams track-group presets.",
        "No saxophone, clarinet, flute, trumpet, brass, violin, strings, choir, or organ.",
        "",
    ]

    for name, track_events in events.items():
        if not track_events:
            continue
        bank, preset, source = PRESETS[name]
        selected.append(f"{name}: bank={bank} preset={preset} | {source}")
        stem = render_stem(args.soundfont, bank, preset, track_events)
        length = min(total, len(stem))
        context[:length] += stem[:length] * context_gains.get(name, 0.0)
        if name in melody_gains:
            melody[:length] += stem[:length] * melody_gains[name]

    context = np.tanh(context * 1.03)
    melody = np.tanh(melody * 1.00)

    melody_wav = args.out / "KRIS_NEW_FROM_ZERO_8BAR_MELODY.wav"
    context_wav = args.out / "KRIS_NEW_FROM_ZERO_8BAR_CONTEXT.wav"
    write_wav(melody_wav, melody)
    write_wav(context_wav, context)
    encode_mp3(melody_wav, args.out / "KRIS_NEW_FROM_ZERO_8BAR_MELODY.mp3")
    encode_mp3(context_wav, args.out / "KRIS_NEW_FROM_ZERO_8BAR_CONTEXT.mp3")
    export_midi(args.out / "KRIS_NEW_FROM_ZERO_8BAR.mid", events)
    (args.out / "KRIS_NEW_FROM_ZERO_8BAR_INSTRUMENTS.txt").write_text(
        "\n".join(selected) + "\n", encoding="utf-8"
    )
    print("\n".join(selected))


if __name__ == "__main__":
    main()
