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
BPM = 150
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

# Only the two requested headline lead timbres plus minimal rhythm support.
PRESETS = {
    "MEGALOVANIA Main Lead": (0, 0, "100 MEGALOVANIA - Overdriven Guitar"),
    "Hopes Main Lead": (3, 8, "087 Hopes and Dreams - Lead Guitar"),
    "MEGALOVANIA Bass": (0, 14, "100 MEGALOVANIA - Bass"),
    "Hopes POWER Drums": (2, 127, "087 Hopes and Dreams - POWER DrumKit"),
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


def build_events() -> Dict[str, List[Event]]:
    events = {name: [] for name in PRESETS}

    # Completely new D-minor theme. Eight written bars, no random note generation.
    melody: List[Pattern] = [
        [("D4", .5), (None, .25), ("F4", .25), ("G4", .5), ("A4", 1.0),
         ("G4", .5), ("F4", .5), ("D4", .5)],
        [("C4", .5), ("D4", .5), ("F4", 1.0), ("Eb4", .5),
         ("D4", .5), ("C4", 1.0)],
        [("Bb3", .5), ("D4", .5), ("F4", .75), ("G4", .25),
         ("F4", .5), ("D4", .5), ("C4", 1.0)],
        [("A3", .5), ("C#4", .5), ("E4", .5), ("G4", .5),
         ("F4", .5), ("E4", .5), ("D4", 1.0)],
        [("D4", .5), ("A4", .5), ("G4", .5), ("F4", .5),
         ("D4", .5), ("F4", .5), ("A4", 1.0)],
        [("Bb4", .5), ("A4", .5), ("G4", 1.0), ("F4", .5),
         ("E4", .5), ("D4", 1.0)],
        [("F4", .5), ("G4", .5), ("A4", 1.0), ("C5", .5),
         ("Bb4", .5), ("A4", 1.0)],
        [("G4", .5), ("F4", .5), ("E4", .5), ("C#4", .5),
         ("D4", 2.0)],
    ]

    roots = ["D2", "Bb1", "F2", "A1", "D2", "Bb1", "G1", "A1"]

    for bar, phrase in enumerate(melody):
        start = bar * 4.0

        # Bars 1-2: MEGALOVANIA lead alone.
        # Bars 3-4: Hopes and Dreams lead alone.
        # Bars 5-8: both headline leads together.
        if bar < 2:
            add_pattern(events, "MEGALOVANIA Main Lead", start, phrase, 118)
        elif bar < 4:
            add_pattern(events, "Hopes Main Lead", start, phrase, 116)
        else:
            add_pattern(events, "MEGALOVANIA Main Lead", start, phrase, 102)
            add_pattern(events, "Hopes Main Lead", start, phrase, 94)

        root = note_number(roots[bar])
        bass_notes = [root, root, root + 7, root, root + 12, root + 7, root, root + 7]
        for step, pitch in enumerate(bass_notes):
            add_note(events, "MEGALOVANIA Bass", start + step * .5, pitch, .34,
                     94 if step in (0, 4) else 70)

        # Firm rock rhythm, intentionally sparse.
        for pos in (0.0, 2.0):
            add_note(events, "Hopes POWER Drums", start + pos, 36, .08, 112)
        for pos in (1.0, 3.0):
            add_note(events, "Hopes POWER Drums", start + pos, 38, .08, 118)
        for step in range(8):
            add_note(events, "Hopes POWER Drums", start + step * .5, 42, .04,
                     48 if step % 2 else 60)
        if bar in (3, 7):
            for off, drum in ((3.0, 45), (3.25, 47), (3.5, 48), (3.75, 50)):
                add_note(events, "Hopes POWER Drums", start + off, drum, .06, 86)

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
    synth.set_reverb(roomsize=.10, damping=.72, width=.62, level=.05)
    synth.set_chorus(nr=2, level=.08, speed=.22, depth=1.4, type=0)

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
        "-af", "highpass=f=35,lowpass=f=12000,acompressor=threshold=-18dB:ratio=2:attack=8:release=110,alimiter=limit=0.96",
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
        "MEGALOVANIA Main Lead": .82,
        "Hopes Main Lead": .78,
        "MEGALOVANIA Bass": .58,
        "Hopes POWER Drums": .62,
    }
    melody_gains = {
        "MEGALOVANIA Main Lead": .92,
        "Hopes Main Lead": .92,
    }

    selected = [
        f"BPM: {BPM}",
        "Key: D minor",
        "Only two headline lead timbres: MEGALOVANIA Overdriven Guitar and Hopes and Dreams Lead Guitar.",
        "No piano, square, pulse, organ, saxophone, clarinet, flute, trumpet, brass, violin, strings, or choir.",
        "Bass and POWER drums are rhythm support only.",
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

    context = np.tanh(context * 1.02)
    melody = np.tanh(melody * 1.00)

    melody_wav = args.out / "KRIS_TWO_MAIN_LEADS_MELODY.wav"
    context_wav = args.out / "KRIS_TWO_MAIN_LEADS_CONTEXT.wav"
    write_wav(melody_wav, melody)
    write_wav(context_wav, context)
    encode_mp3(melody_wav, args.out / "KRIS_TWO_MAIN_LEADS_MELODY.mp3")
    encode_mp3(context_wav, args.out / "KRIS_TWO_MAIN_LEADS_CONTEXT.mp3")
    export_midi(args.out / "KRIS_TWO_MAIN_LEADS.mid", events)
    (args.out / "KRIS_TWO_MAIN_LEADS_INSTRUMENTS.txt").write_text(
        "\n".join(selected) + "\n", encoding="utf-8"
    )
    print("\n".join(selected))


if __name__ == "__main__":
    main()
