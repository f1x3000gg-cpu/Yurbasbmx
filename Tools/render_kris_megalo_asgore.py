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

PRESETS = {
    "MEGALOVANIA Main Guitar": (0, 0, "100 MEGALOVANIA - Overdriven Guitar"),
    "MEGALOVANIA Impact": (0, 9, "100 MEGALOVANIA - Impact Hit"),
    "MEGALOVANIA Bass": (0, 14, "100 MEGALOVANIA - Bass"),
    "ASGORE Piano": (2, 64, "077 ASGORE - Piano"),
    "ASGORE Timpani": (2, 74, "077 ASGORE - Timpani"),
    "ASGORE Drums": (2, 80, "077 ASGORE - Drumkit"),
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
    pitch = note if isinstance(note, int) else note_number(note)
    events[track].append((float(start), float(duration), int(pitch), int(velocity)))


def add_pattern(events: Dict[str, List[Event]], track: str, start: float,
                pattern: Pattern, velocity: int, transpose: int = 0) -> None:
    cursor = start
    for note, duration in pattern:
        if note is not None:
            add_note(events, track, cursor, note_number(note) + transpose,
                     max(0.05, duration * 0.90), velocity)
        cursor += duration


def build_events() -> tuple[Dict[str, List[Event]], Dict[str, List[Event]]]:
    melody_only = {name: [] for name in PRESETS}
    context = {name: [] for name in PRESETS}

    # New eight-bar D-minor battle hook. The MEGALOVANIA overdriven guitar is
    # the only melodic lead. ASGORE piano is used as dark rhythmic support and
    # short replies, not as a second lead voice.
    melody: List[Pattern] = [
        [("D4", .5), ("D4", .5), ("F4", .5), ("A4", .5), ("G4", 1.0), ("F4", 1.0)],
        [("D4", .5), ("D4", .5), ("C5", .5), ("Bb4", .5), ("A4", 1.0), ("G4", 1.0)],
        [("F4", .5), ("F4", .5), ("A4", .5), ("C5", .5), ("Bb4", 1.0), ("A4", 1.0)],
        [("E4", .5), ("G4", .5), ("A4", .5), ("C#5", .5), ("D5", 2.0)],
        [("A4", .5), ("A4", .5), ("C5", .5), ("D5", .5), ("F5", 1.0), ("E5", 1.0)],
        [("D5", .5), ("C5", .5), ("A4", .5), ("Bb4", .5), ("A4", 1.0), ("G4", 1.0)],
        [("F4", .5), ("A4", .5), ("D5", .5), ("C5", .5), ("Bb4", 1.0), ("A4", 1.0)],
        [("E4", .5), ("F4", .5), ("G4", .5), ("C#5", .5), ("D5", 2.0)],
    ]

    roots = ["D2", "Bb1", "F2", "A1", "D2", "C2", "Bb1", "A1"]
    piano_chords = [
        ["D2", "A2", "D3", "F3"],
        ["Bb1", "F2", "Bb2", "D3"],
        ["F2", "C3", "F3", "A3"],
        ["A1", "E2", "A2", "C#3"],
        ["D2", "A2", "D3", "F3"],
        ["C2", "G2", "C3", "E3"],
        ["Bb1", "F2", "Bb2", "D3"],
        ["A1", "E2", "A2", "C#3"],
    ]

    for bar, phrase in enumerate(melody):
        start = bar * 4.0
        add_pattern(melody_only, "MEGALOVANIA Main Guitar", start, phrase, 116)
        add_pattern(context, "MEGALOVANIA Main Guitar", start, phrase, 116)

        # ASGORE piano: heavy low chord attacks and a brief response at the end
        # of every second bar. It never replaces the guitar lead.
        chord = piano_chords[bar]
        for pos, vel in ((0.0, 88), (2.0, 72)):
            for note in chord:
                add_note(context, "ASGORE Piano", start + pos, note, .58, vel)
        if bar in (1, 3, 5, 7):
            response = [("A3", .5), ("C4", .5), ("D4", 1.0)] if bar < 4 else [("D4", .5), ("F4", .5), ("A4", 1.0)]
            add_pattern(context, "ASGORE Piano", start + 2.0, response, 64)

        root = note_number(roots[bar])
        bass_pattern = [root, root + 7, root + 12, root + 7,
                        root, root + 7, root + 12, root + 7]
        for step, pitch in enumerate(bass_pattern):
            add_note(context, "MEGALOVANIA Bass", start + step * .5,
                     pitch, .34, 98 if step in (0, 4) else 76)

        # ASGORE drumkit: direct rock beat, no note spam.
        for pos in (0.0, 2.0):
            add_note(context, "ASGORE Drums", start + pos, 36, .08, 112)
        for pos in (1.0, 3.0):
            add_note(context, "ASGORE Drums", start + pos, 38, .08, 118)
        for step in range(8):
            add_note(context, "ASGORE Drums", start + step * .5, 42, .04,
                     54 if step % 2 else 64)
        if bar in (3, 7):
            for off, drum in ((3.0, 45), (3.25, 47), (3.5, 48), (3.75, 50)):
                add_note(context, "ASGORE Drums", start + off, drum, .05, 82)

        # Timpani and impact mark the large phrases.
        if bar in (0, 3, 4, 7):
            add_note(context, "ASGORE Timpani", start, root + 12, .42, 92)
        if bar in (0, 4, 7):
            add_note(context, "MEGALOVANIA Impact", start, root + 24, .18, 96)

    return melody_only, context


def render_stem(soundfont: Path, bank: int, preset: int,
                events: Sequence[Event], gain: float = .82) -> np.ndarray:
    total_frames = int(((BARS * 4 * BEAT_SECONDS) + TAIL_SECONDS) * SR)
    synth = fluidsynth.Synth(gain=gain, samplerate=SR)
    sfid = synth.sfload(str(soundfont))
    result = synth.program_select(0, sfid, bank, preset)
    if result != 0:
        synth.delete()
        raise RuntimeError(f"Cannot select bank={bank}, preset={preset}")
    synth.set_reverb(roomsize=.14, damping=.72, width=.72, level=.07)
    synth.set_chorus(nr=2, level=.08, speed=.25, depth=1.4, type=0)

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


def mix(soundfont: Path, events: Dict[str, List[Event]]) -> np.ndarray:
    total_frames = int(((BARS * 4 * BEAT_SECONDS) + TAIL_SECONDS) * SR)
    result = np.zeros((total_frames, 2), dtype=np.float64)
    gains = {
        "MEGALOVANIA Main Guitar": .92,
        "MEGALOVANIA Impact": .42,
        "MEGALOVANIA Bass": .58,
        "ASGORE Piano": .57,
        "ASGORE Timpani": .50,
        "ASGORE Drums": .69,
    }
    for name, track_events in events.items():
        if not track_events:
            continue
        bank, preset, _ = PRESETS[name]
        stem = render_stem(soundfont, bank, preset, track_events)
        result[:len(stem)] += stem * gains[name]
    peak = float(np.max(np.abs(result)))
    if peak > 0:
        result = result / peak * .93
    return result


def write_wav(path: Path, audio: np.ndarray) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes(pcm.tobytes())


def encode_mp3(wav_path: Path, mp3_path: Path) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path),
        "-af", "highpass=f=30,lowpass=f=12500,acompressor=threshold=-18dB:ratio=2:attack=8:release=120,alimiter=limit=0.96",
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
        track.append(Message("control_change", channel=midi_channel, control=0,
                             value=min(127, bank), time=0))
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
    parser.add_argument("--soundfont", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    melody_events, context_events = build_events()
    melody_audio = mix(args.soundfont, melody_events)
    context_audio = mix(args.soundfont, context_events)

    melody_wav = args.out / "KRIS_MEGALO_ASGORE_MELODY.wav"
    context_wav = args.out / "KRIS_MEGALO_ASGORE_CONTEXT.wav"
    write_wav(melody_wav, melody_audio)
    write_wav(context_wav, context_audio)
    encode_mp3(melody_wav, args.out / "KRIS_MEGALO_ASGORE_MELODY.mp3")
    encode_mp3(context_wav, args.out / "KRIS_MEGALO_ASGORE_CONTEXT.mp3")
    export_midi(args.out / "KRIS_MEGALO_ASGORE.mid", context_events)

    lines = [
        f"BPM: {BPM}",
        "Key: D minor",
        "Only melodic lead: 100 MEGALOVANIA - Overdriven Guitar (bank 0, preset 0)",
        "ASGORE element: 077 ASGORE - Piano (bank 2, preset 64), used as dark chords and short replies",
        "Support: 100 MEGALOVANIA - Bass; 077 ASGORE - Timpani and Drumkit; MEGALOVANIA Impact Hit",
        "No Hopes and Dreams Lead Guitar. No saxophone, clarinet, flute, trumpet, violin, strings, choir, or organ.",
        "The melody is original and manually written; it does not copy Undertale notes.",
    ]
    (args.out / "KRIS_MEGALO_ASGORE_INSTRUMENTS.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
