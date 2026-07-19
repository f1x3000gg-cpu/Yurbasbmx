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
BPM = 146
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
    "MEGALOVANIA Mid Guitar": (0, 0, "100 MEGALOVANIA - Overdriven Guitar"),
    "MEGALOVANIA Bass": (0, 14, "100 MEGALOVANIA - Bass"),
    "ASGORE Piano": (2, 64, "077 ASGORE - Piano"),
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
            notes: Pattern, velocity: int) -> None:
    cursor = start
    for note, duration in notes:
        if note is not None:
            add(events, track, cursor, note, max(.06, duration * .88), velocity)
        cursor += duration


def chord(events: Dict[str, List[Event]], track: str, start: float,
          notes: Sequence[str], duration: float, velocity: int) -> None:
    for note in notes:
        add(events, track, start, note, duration, velocity)


def build_events() -> tuple[Dict[str, List[Event]], Dict[str, List[Event]], Dict[str, List[Event]]]:
    guitar_only = {name: [] for name in PRESETS}
    piano_only = {name: [] for name in PRESETS}
    context = {name: [] for name in PRESETS}

    # Same original hook in a middle register: not high/tense and not low/rough.
    melody: List[Pattern] = [
        [("A3", .5), ("C4", .5), ("D4", 1.0), ("F4", .5), ("E4", .5), ("D4", 1.0)],
        [("C4", .5), ("A3", .5), ("G3", 1.0), ("A3", .5), ("C4", .5), ("D4", 1.0)],
        [("F4", .5), ("A4", .5), ("G4", 1.0), ("F4", .5), ("E4", .5), ("C4", 1.0)],
        [("D4", .5), ("C4", .5), ("A3", 1.0), ("C#4", .5), ("E4", .5), ("D4", 1.0)],
        [("D4", .5), ("F4", .5), ("A4", 1.0), ("G4", .5), ("F4", .5), ("E4", 1.0)],
        [("C4", .5), ("D4", .5), ("F4", 1.0), ("E4", .5), ("D4", .5), ("A3", 1.0)],
        [("Bb3", .5), ("D4", .5), ("F4", 1.0), ("E4", .5), ("C#4", .5), ("D4", 1.0)],
        [("A3", .5), ("C#4", .5), ("E4", 1.0), ("D4", 2.0)],
    ]

    roots = ["D2", "Bb1", "F2", "A1", "D2", "C2", "Bb1", "A1"]
    piano_chords = [
        ["D3", "A3", "D4", "F4"],
        ["Bb2", "F3", "Bb3", "D4"],
        ["F3", "C4", "F4", "A4"],
        ["A2", "E3", "A3", "C#4"],
        ["D3", "A3", "D4", "F4"],
        ["C3", "G3", "C4", "E4"],
        ["Bb2", "F3", "Bb3", "D4"],
        ["A2", "E3", "A3", "C#4"],
    ]

    # Separate 4-bar proofs.
    for bar in range(4):
        pattern(guitar_only, "MEGALOVANIA Mid Guitar", bar * 4.0, melody[bar], 84)
        pattern(piano_only, "ASGORE Piano", bar * 4.0, melody[bar + 4], 110)

    for bar in range(BARS):
        start = bar * 4.0

        if bar < 4:
            # Guitar states the hook quietly in the middle register.
            pattern(context, "MEGALOVANIA Mid Guitar", start, melody[bar], 84)
        else:
            # The actual ASGORE piano becomes the clear main lead.
            pattern(context, "ASGORE Piano", start, melody[bar], 112)
            # Very small guitar answer, same register, never an octave down.
            if bar in (5, 7):
                reply: Pattern = [("A3", .5), ("C4", .5), ("D4", 1.0), (None, 2.0)]
                pattern(context, "MEGALOVANIA Mid Guitar", start + 2.0, reply, 42)

        # ASGORE piano harmony is clearly audible in all bars, but does not bury the lead.
        for pos, vel in ((0.0, 64), (2.0, 52)):
            chord(context, "ASGORE Piano", start + pos, piano_chords[bar], .42, vel)

        # Minimal bass: no heavy left-side rumble.
        root = nn(roots[bar])
        add(context, "MEGALOVANIA Bass", start, root, .65, 60)
        add(context, "MEGALOVANIA Bass", start + 2.0, root + 7, .55, 48)

        # Straight ASGORE battle groove.
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

        if bar in (0, 4, 7):
            add(context, "ASGORE Timpani", start, root + 12, .40, 76)

    return guitar_only, piano_only, context


def render_stem(soundfont: Path, bank: int, preset: int,
                events: Sequence[Event]) -> np.ndarray:
    total = int(((BARS * 4 * BEAT) + TAIL) * SR)
    synth = fluidsynth.Synth(gain=.76, samplerate=SR)
    sfid = synth.sfload(str(soundfont))
    if synth.program_select(0, sfid, bank, preset) != 0:
        synth.delete()
        raise RuntimeError(f"Cannot select bank={bank} preset={preset}")
    synth.set_reverb(roomsize=.10, damping=.74, width=.64, level=.045)
    synth.set_chorus(nr=2, level=.045, speed=.20, depth=1.0, type=0)

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


def mix(soundfont: Path, events: Dict[str, List[Event]], proof: str = "context") -> np.ndarray:
    total = int(((BARS * 4 * BEAT) + TAIL) * SR)
    output = np.zeros((total, 2), dtype=np.float64)
    gains = {
        "MEGALOVANIA Mid Guitar": .52 if proof == "context" else .78,
        "MEGALOVANIA Bass": .30,
        "ASGORE Piano": .82 if proof == "context" else .92,
        "ASGORE Timpani": .38,
        "ASGORE Drums": .58,
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
        "-af", "highpass=f=55,lowpass=f=11800,acompressor=threshold=-18dB:ratio=1.8:attack=10:release=130,alimiter=limit=.96",
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

    guitar_events, piano_events, context_events = build_events()
    variants = [
        ("KRIS_MID_MEGALOVANIA_4BAR", guitar_events, "guitar"),
        ("KRIS_EXACT_ASGORE_PIANO_4BAR", piano_events, "piano"),
        ("KRIS_MID_MEGALO_ASGORE_PIANO_CONTEXT", context_events, "context"),
    ]
    for stem, events, proof in variants:
        audio = mix(args.soundfont, events, proof)
        wav_path = args.out / f"{stem}.wav"
        write_wav(wav_path, audio)
        encode(wav_path, args.out / f"{stem}.mp3")

    export_midi(args.out / "KRIS_MID_MEGALO_ASGORE_PIANO.mid", context_events)
    report = [
        f"BPM: {BPM}",
        "MEGALOVANIA lead range: G3-A4, deliberately middle register.",
        "ASGORE main second sound: 077 ASGORE - Piano, bank 2 preset 64.",
        "No ASGORE Brass, guitar, flute, saxophone, clarinet, violin, strings, or choir.",
        "ASGORE Timpani and Drumkit are rhythm support only.",
        "The ASGORE piano is louder than the MEGALOVANIA guitar in the combined proof.",
    ]
    (args.out / "KRIS_MID_MEGALO_ASGORE_PIANO_INSTRUMENTS.txt").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
