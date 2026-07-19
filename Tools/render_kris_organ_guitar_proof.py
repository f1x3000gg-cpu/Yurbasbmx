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
BPM = 156
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

# Only track-group presets from MEGALOVANIA and Hopes and Dreams.
# No saxophone, clarinet, flute, trumpet, brass, violin, strings, choir, or organ.
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
    cursor = start
    for note, duration in pattern:
        if note is not None:
            add_note(events, track, cursor, note_number(note) + transpose,
                     duration * 0.93, velocity)
        cursor += duration


def add_chord(events: Dict[str, List[Event]], track: str, start: float,
              notes: Sequence[str], duration: float, velocity: int) -> None:
    for note in notes:
        add_note(events, track, start, note, duration, velocity)


def build_events() -> Dict[str, List[Event]]:
    events = {name: [] for name in PRESETS}

    # One deliberately written D-minor melody. It is eight bars long and uses
    # repeated phrases, answers, held notes, and a clear cadence. No randomizer.
    melody: List[Pattern] = [
        [("A4", .5), ("C5", .5), ("D5", 1), ("F5", .5), ("E5", .5), ("D5", 1)],
        [("C5", .5), ("A4", .5), ("G4", 1), ("A4", .5), ("C5", .5), ("D5", 1)],
        [("F5", .5), ("A5", .5), ("G5", 1), ("F5", .5), ("E5", .5), ("C5", 1)],
        [("D5", 1), ("C5", .5), ("A4", .5), ("G#4", .5), ("B4", .5), ("A4", 1)],
        [("A4", .5), ("D5", .5), ("F5", 1), ("A5", .5), ("G5", .5), ("F5", 1)],
        [("E5", .5), ("D5", .5), ("C5", 1), ("A4", .5), ("C5", .5), ("E5", 1)],
        [("F5", 1), ("E5", .5), ("D5", .5), ("C5", 1), ("A4", 1)],
        [("Bb4", .5), ("C5", .5), ("D5", 1), ("C#5", .5), ("E5", .5), ("D5", 1)],
    ]

    chords = [
        ["D3", "A3", "C4", "F4"],
        ["Bb2", "F3", "A3", "D4"],
        ["F2", "C3", "E3", "A3"],
        ["C3", "G3", "Bb3", "E4"],
        ["D3", "A3", "C4", "F4"],
        ["A2", "E3", "G3", "C4"],
        ["Bb2", "F3", "A3", "D4"],
        ["A2", "E3", "G3", "C#4"],
    ]
    roots = ["D2", "Bb1", "F2", "C2", "D2", "A1", "Bb1", "A1"]

    for bar in range(BARS):
        start = bar * 4.0
        # Lead is guitar, never a wind instrument. The square only reinforces
        # selected phrase peaks instead of replacing the melody.
        add_pattern(events, "Hopes Lead Guitar", start, melody[bar], 108)
        add_pattern(events, "Megalo Over Guitar", start, melody[bar], 58, transpose=-12)
        if bar in (2, 4, 7):
            add_pattern(events, "Megalo Square25", start, melody[bar], 35, transpose=-12)
        else:
            add_pattern(events, "Hopes Pulse25", start, melody[bar], 29, transpose=-12)

        # Piano states harmony twice per bar, simple enough to leave space.
        add_chord(events, "Hopes Piano1", start, chords[bar], .62, 68)
        add_chord(events, "Hopes Piano1", start + 2.0, chords[bar], .62, 58)

        # Bass is a repeated root-fifth figure, not random movement.
        root = note_number(roots[bar])
        bass_seq = [root, root + 7, root + 12, root + 7,
                    root, root + 7, root + 12, root + 7]
        for step, pitch in enumerate(bass_seq):
            add_note(events, "Megalo Bass", start + step * .5, pitch, .36,
                     96 if step in (0, 4) else 76)

        # Background guitar uses syncopated power-chord stabs.
        guitar_root = root + 12
        power = [guitar_root, guitar_root + 7, guitar_root + 12]
        for pos in (0.0, 1.5, 2.0, 3.5):
            for pitch in power:
                add_note(events, "Megalo Background Guitar", start + pos,
                         pitch, .27, 67 if pos in (0.0, 2.0) else 54)

        # POWER drums: firm rock pattern, no breakcore spam.
        for pos in (0.0, 2.0):
            add_note(events, "Hopes POWER", start + pos, 36, .08, 108)
        for pos in (1.0, 3.0):
            add_note(events, "Hopes POWER", start + pos, 38, .09, 116)
        for step in range(8):
            add_note(events, "Hopes POWER", start + step * .5, 42, .045,
                     49 if step % 2 else 61)
        if bar in (3, 7):
            for off, drum in ((3.0, 45), (3.25, 47), (3.5, 48), (3.75, 50)):
                add_note(events, "Hopes POWER", start + off, drum, .06, 82)
        if bar in (0, 4, 7):
            add_chord(events, "Megalo Impact", start,
                      [chords[bar][0], chords[bar][2]], .20, 92)

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
    synth.set_reverb(roomsize=.13, damping=.68, width=.68, level=.07)
    synth.set_chorus(nr=2, level=.11, speed=.25, depth=1.7, type=0)

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
    total_frames = int(((BARS * 4 * BEAT_SECONDS) + TAIL_SECONDS) * SR)
    melody_mix = np.zeros((total_frames, 2), dtype=np.float64)
    context_mix = np.zeros_like(melody_mix)

    context_gains = {
        "Megalo Over Guitar": .48,
        "Megalo Square25": .23,
        "Megalo Impact": .42,
        "Megalo Background Guitar": .37,
        "Megalo Bass": .64,
        "Hopes Piano1": .42,
        "Hopes POWER": .67,
        "Hopes Pulse25": .18,
        "Hopes Duty": .0,
        "Hopes Lead Guitar": .79,
    }
    melody_gains = {
        "Hopes Lead Guitar": .90,
        "Hopes Piano1": .30,
        "Hopes Pulse25": .12,
        "Megalo Square25": .12,
    }

    report = [
        f"BPM: {BPM}",
        "Key: D minor",
        "Eight-bar hand-written melody; no random-note generator.",
        "No saxophone, clarinet, flute, trumpet, brass, violin, strings, choir, or organ.",
        "The named presets are track-group presets from the compiled UNDERTALE Soundfont 2; this report does not claim they are the original licensed VST/sample files.",
        "",
    ]

    for name, track_events in events.items():
        if not track_events:
            continue
        bank, preset, source = PRESETS[name]
        report.append(f"{name}: bank={bank} preset={preset} | {source}")
        stem = render_stem(args.soundfont, bank, preset, track_events)
        length = min(total_frames, len(stem))
        context_mix[:length] += stem[:length] * context_gains.get(name, 0.0)
        melody_mix[:length] += stem[:length] * melody_gains.get(name, 0.0)

    context_mix = np.tanh(context_mix * 1.04)
    melody_mix = np.tanh(melody_mix * 1.02)

    melody_wav = args.out / "KRIS_MEGALO_ASRIEL_8BAR_MELODY.wav"
    context_wav = args.out / "KRIS_MEGALO_ASRIEL_8BAR_CONTEXT.wav"
    write_wav(melody_wav, melody_mix)
    write_wav(context_wav, context_mix)
    encode_mp3(melody_wav, args.out / "KRIS_MEGALO_ASRIEL_8BAR_MELODY.mp3")
    encode_mp3(context_wav, args.out / "KRIS_MEGALO_ASRIEL_8BAR_CONTEXT.mp3")
    export_midi(args.out / "KRIS_MEGALO_ASRIEL_8BAR.mid", events)
    (args.out / "KRIS_MEGALO_ASRIEL_8BAR_INSTRUMENTS.txt").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print("\n".join(report))


if __name__ == "__main__":
    main()
