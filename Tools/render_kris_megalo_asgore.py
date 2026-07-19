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
BPM = 144
BEAT = 60.0 / BPM
BARS = 8
TAIL = 2.0
TPB = 480

PITCH = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

Event = Tuple[float, float, int, int]
Pattern = Sequence[Tuple[str | None, float]]

PRESETS = {
    "MEGALOVANIA Original Guitar": (0, 0, "100 MEGALOVANIA - Overdriven Guitar"),
    "ASGORE Orchestra Hit": (2, 77, "077 ASGORE - Orchestra Hit"),
    "ASGORE Timpani": (2, 74, "077 ASGORE - Timpani"),
    "ASGORE Drums": (2, 80, "077 ASGORE - Drumkit"),
}


def nn(note: str) -> int:
    if len(note) >= 3 and note[1] in "#b":
        name, octave = note[:2], int(note[2:])
    else:
        name, octave = note[0], int(note[1:])
    return 12 * (octave + 1) + PITCH[name]


def add(events: Dict[str, List[Event]], track: str, start: float,
        note: str | int, duration: float, velocity: int) -> None:
    pitch = note if isinstance(note, int) else nn(note)
    events[track].append((float(start), float(duration), int(pitch), int(velocity)))


def pattern(events: Dict[str, List[Event]], track: str, start: float,
            notes: Pattern, velocity: int, gate: float = 0.68) -> None:
    cursor = start
    for note, duration in notes:
        if note is not None:
            add(events, track, cursor, note, max(0.05, duration * gate), velocity)
        cursor += duration


def melody() -> List[Pattern]:
    # New original battle melody. Deliberately limited to C4-A4: a normal middle register.
    return [
        [("D4", .5), ("F4", .5), ("A4", .75), ("G4", .25), ("F4", .5), ("E4", .5), ("D4", 1.0)],
        [("F4", .5), ("E4", .5), ("C4", .75), ("D4", .25), ("F4", .5), ("G4", .5), ("E4", 1.0)],
        [("D4", .5), ("E4", .5), ("F4", .5), ("A4", .5), ("G4", .75), ("F4", .25), ("D4", 1.0)],
        [("C4", .5), ("D4", .5), ("F4", .75), ("E4", .25), ("D4", .5), ("C#4", .5), ("D4", 1.0)],
        [("F4", .5), ("A4", .5), ("G4", .75), ("F4", .25), ("E4", .5), ("D4", .5), ("F4", 1.0)],
        [("E4", .5), ("G4", .5), ("F4", .75), ("E4", .25), ("D4", .5), ("C4", .5), ("E4", 1.0)],
        [("D4", .5), ("F4", .5), ("A4", .5), ("G4", .5), ("F4", .5), ("E4", .5), ("D4", 1.0)],
        [("C4", .5), ("D4", .5), ("E4", .5), ("F4", .5), ("E4", .5), ("C#4", .5), ("D4", 1.0)],
    ]


def build_events() -> tuple[Dict[str, List[Event]], Dict[str, List[Event]]]:
    guitar_only = {name: [] for name in PRESETS}
    context = {name: [] for name in PRESETS}
    hook = melody()

    # Isolated guitar proof: exact preset, no EQ, no pitch shift, no doubling.
    for bar in range(4):
        pattern(guitar_only, "MEGALOVANIA Original Guitar", bar * 4.0,
                hook[bar], velocity=70, gate=0.66)

    for bar in range(BARS):
        start = bar * 4.0
        pattern(context, "MEGALOVANIA Original Guitar", start,
                hook[bar], velocity=70, gate=0.66)

        # ASGORE identity without piano, pulse, brass, strings or wind lead.
        if bar in (0, 2, 4, 6):
            add(context, "ASGORE Orchestra Hit", start, "D4", .22, 58)
        if bar in (3, 7):
            add(context, "ASGORE Orchestra Hit", start + 3.5, "A3", .20, 64)

        # Timpani reinforces the downbeat without adding a bass-guitar layer.
        root = ("D3", "C3", "Bb2", "A2", "D3", "C3", "Bb2", "A2")[bar]
        add(context, "ASGORE Timpani", start, root, .32, 62)
        if bar in (3, 7):
            add(context, "ASGORE Timpani", start + 3.5, root, .25, 58)

        # Keep the approved ASGORE drum groove.
        for pos in (0.0, 2.0):
            add(context, "ASGORE Drums", start + pos, 36, .07, 96)
        for pos in (1.0, 3.0):
            add(context, "ASGORE Drums", start + pos, 38, .08, 108)
        for step in range(8):
            add(context, "ASGORE Drums", start + step * .5, 42, .04,
                42 if step % 2 else 52)
        if bar in (3, 7):
            for off, drum in ((3.0, 45), (3.25, 47), (3.5, 48), (3.75, 50)):
                add(context, "ASGORE Drums", start + off, drum, .05, 72)

    return guitar_only, context


def render_stem(soundfont: Path, bank: int, preset: int,
                events: Sequence[Event]) -> np.ndarray:
    total = int(((BARS * 4 * BEAT) + TAIL) * SR)
    synth = fluidsynth.Synth(gain=.72, samplerate=SR)
    sfid = synth.sfload(str(soundfont))
    if synth.program_select(0, sfid, bank, preset) != 0:
        synth.delete()
        raise RuntimeError(f"Cannot select bank={bank} preset={preset}")

    # Small common room only. No EQ, filtering or tone shaping on the guitar.
    synth.set_reverb(roomsize=.07, damping=.78, width=.58, level=.025)
    synth.set_chorus(nr=2, level=.018, speed=.18, depth=.55, type=0)

    timeline = []
    for start, duration, note, velocity in events:
        timeline.append((int(start * BEAT * SR), 1, note, velocity))
        timeline.append((int((start + duration) * BEAT * SR), 0, note, 0))
    timeline.sort(key=lambda item: (item[0], item[1]))

    chunks = []
    cursor = 0
    for frame, on, note, velocity in timeline:
        if frame > cursor:
            chunks.append(synth.get_samples(frame - cursor))
            cursor = frame
        if on:
            synth.noteon(0, note, velocity)
        else:
            synth.noteoff(0, note)
    if cursor < total:
        chunks.append(synth.get_samples(total - cursor))
    synth.delete()

    if not chunks:
        return np.zeros((total, 2), dtype=np.float64)
    raw = np.concatenate(chunks).astype(np.float64) / 32768.0
    return raw.reshape(-1, 2)


def mix(soundfont: Path, events: Dict[str, List[Event]], proof: str) -> np.ndarray:
    total = int(((BARS * 4 * BEAT) + TAIL) * SR)
    output = np.zeros((total, 2), dtype=np.float64)
    gains = {
        "MEGALOVANIA Original Guitar": .72 if proof == "context" else .88,
        "ASGORE Orchestra Hit": .34,
        "ASGORE Timpani": .35,
        "ASGORE Drums": .59,
    }

    for name, track_events in events.items():
        if not track_events:
            continue
        bank, preset, _ = PRESETS[name]
        stem = render_stem(soundfont, bank, preset, track_events)
        output[:len(stem)] += stem * gains[name]

    peak = float(np.max(np.abs(output)))
    if peak > 0:
        output = output / peak * .93
    return output


def write_wav(path: Path, audio: np.ndarray) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes(pcm.tobytes())


def encode(wav_path: Path, mp3_path: Path) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path),
        "-codec:a", "libmp3lame", "-q:a", "2", str(mp3_path),
    ], check=True)


def export_midi(path: Path, events: Dict[str, List[Event]]) -> None:
    midi = MidiFile(ticks_per_beat=TPB)
    tempo = MidiTrack()
    tempo.append(MetaMessage("set_tempo", tempo=bpm2tempo(BPM), time=0))
    tempo.append(MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    midi.tracks.append(tempo)

    channel = 0
    for name, track_events in events.items():
        if not track_events:
            continue
        while channel == 9:
            channel += 1
        ch = channel % 16
        channel += 1
        bank, preset, _ = PRESETS[name]
        track = MidiTrack()
        track.append(MetaMessage("track_name", name=name, time=0))
        track.append(Message("control_change", channel=ch, control=0, value=bank, time=0))
        track.append(Message("program_change", channel=ch, program=preset, time=0))

        timeline = []
        for start, duration, note, velocity in track_events:
            timeline.append((round(start * TPB), 1, note, velocity))
            timeline.append((round((start + duration) * TPB), 0, note, 0))
        timeline.sort(key=lambda item: (item[0], item[1]))

        previous = 0
        for tick, on, note, velocity in timeline:
            track.append(Message("note_on" if on else "note_off", channel=ch,
                                 note=note, velocity=velocity if on else 0,
                                 time=tick - previous))
            previous = tick
        midi.tracks.append(track)
    midi.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--soundfont", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    guitar_only, context = build_events()
    guitar_audio = mix(args.soundfont, guitar_only, "guitar")
    context_audio = mix(args.soundfont, context, "context")

    guitar_wav = args.out / "KRIS_NORMAL_MEGALOVANIA_GUITAR_4BAR.wav"
    context_wav = args.out / "KRIS_NORMAL_MEGALO_ASGORE_CONTEXT.wav"
    write_wav(guitar_wav, guitar_audio)
    write_wav(context_wav, context_audio)
    encode(guitar_wav, args.out / "KRIS_NORMAL_MEGALOVANIA_GUITAR_4BAR.mp3")
    encode(context_wav, args.out / "KRIS_NORMAL_MEGALO_ASGORE_CONTEXT.mp3")
    export_midi(args.out / "KRIS_NORMAL_MEGALO_ASGORE.mid", context)

    report = [
        "144 BPM, D minor, entirely new melody.",
        "Main lead: 100 MEGALOVANIA - Overdriven Guitar, bank 0 preset 0.",
        "Guitar range: C4-A4 (normal middle register).",
        "The guitar preset is unfiltered and un-EQ'd; no pitch shifting or octave doubling.",
        "ASGORE support: Orchestra Hit, Timpani and Drumkit.",
        "No ASGORE Piano, Pulse/pixel lead, Brass, Strings, Choir or bass-guitar layer.",
        "The guitar remains the only continuous melodic lead.",
    ]
    (args.out / "KRIS_NORMAL_MEGALO_ASGORE_INSTRUMENTS.txt").write_text(
        "\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
