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
BEAT = 60.0 / BPM
BARS = 8
BAR_SECONDS = 4.0 * BEAT
TAIL_SECONDS = 2.2
TPB = 480

PITCH = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

Event = Tuple[float, float, int, int]
Pattern = Sequence[Tuple[str | int | None, float]]

# Exact track-group locations inside the compiled UNDERTALE Soundfont 2 bank.
# Only MEGALOVANIA and Hopes and Dreams palette entries are active.
PRESETS = {
    "Megalo Over Guitar": (0, 0, "100 MEGALOVANIA - Overdriven Guitar"),
    "Megalo Square25":    (0, 8, "100 MEGALOVANIA - 25% Square"),
    "Megalo Impact":      (0, 9, "100 MEGALOVANIA - Impact Hit"),
    "Megalo Bg Guitar":   (0, 13, "100 MEGALOVANIA - Background Guitar"),
    "Megalo Bass":        (0, 14, "100 MEGALOVANIA - Bass"),
    "Hopes Piano1":       (2, 126, "087 Hopes and Dreams - Piano 1"),
    "Hopes POWER":        (2, 127, "087 Hopes and Dreams - POWER DrumKit"),
    "Hopes Pulse25":      (3, 5, "087 Hopes and Dreams - Pulse 25%"),
    "Hopes Duty":         (3, 7, "087 Hopes and Dreams - Duty Cycle"),
    "Hopes Lead Guitar":  (3, 8, "087 Hopes and Dreams - Lead Guitar"),
}


def nn(note: str | int) -> int:
    if isinstance(note, int):
        return note
    if len(note) >= 3 and note[1] in "#b":
        name, octave = note[:2], int(note[2:])
    else:
        name, octave = note[0], int(note[1:])
    return 12 * (octave + 1) + PITCH[name]


def add_note(events: Dict[str, List[Event]], track: str, start: float,
             note: str | int, duration: float, velocity: int) -> None:
    events[track].append((float(start), float(duration), nn(note), int(velocity)))


def add_pattern(events: Dict[str, List[Event]], track: str, start: float,
                pattern: Pattern, velocity: int, shift: int = 0) -> None:
    cursor = float(start)
    for note, duration in pattern:
        if note is not None:
            add_note(events, track, cursor, nn(note) + shift, duration * 0.93, velocity)
        cursor += duration


def add_chord(events: Dict[str, List[Event]], track: str, start: float,
              notes: Sequence[str | int], duration: float, velocity: int) -> None:
    for note in notes:
        add_note(events, track, start, note, duration, velocity)


def make_events() -> Dict[str, List[Event]]:
    events = {name: [] for name in PRESETS}

    # Completely new E-minor phrase. Written as a singable 4-bar call and
    # 4-bar answer before any arrangement was added. No random generation.
    melody: List[Pattern] = [
        [(None, .5), ("B4", .5), ("E5", 1), ("G5", .5), ("F#5", .5), ("E5", 1)],
        [("D5", .5), ("B4", .5), ("A4", 1), ("B4", .5), ("D5", .5), ("E5", 1)],
        [("B4", .5), ("E5", .5), ("G5", 1), ("F#5", .5), ("E5", .5), ("D5", 1)],
        [("B4", .5), ("A4", .5), ("G4", 1), ("A4", .5), ("B4", .5), ("D#5", 1)],
        [("E5", 1), ("B4", .5), ("D5", .5), ("G5", 1), ("F#5", 1)],
        [("E5", .5), ("D5", .5), ("C5", 1), ("B4", .5), ("A4", .5), ("G4", 1)],
        [("A4", .5), ("C5", .5), ("E5", 1), ("D5", .5), ("C5", .5), ("B4", 1)],
        [("G4", .5), ("A4", .5), ("B4", 1), ("D#5", .5), ("E5", .5), ("B4", 1)],
    ]

    chords = [
        ["E3", "B3", "E4", "G4"],
        ["C3", "G3", "C4", "E4"],
        ["G2", "D3", "G3", "B3"],
        ["D3", "A3", "D4", "F#4"],
        ["E3", "B3", "E4", "G4"],
        ["C3", "G3", "C4", "E4"],
        ["A2", "E3", "A3", "C4"],
        ["B2", "F#3", "A3", "D#4"],
    ]
    roots = ["E2", "C2", "G1", "D2", "E2", "C2", "A1", "B1"]

    for bar in range(BARS):
        start = bar * 4.0
        pattern = melody[bar]

        # Lead is guitar, not a wind-like patch. Piano quietly reinforces only
        # the phrase contour. The square sits an octave below, never on top.
        add_pattern(events, "Hopes Lead Guitar", start, pattern, 108)
        add_pattern(events, "Hopes Piano1", start, pattern, 62, shift=-12)
        add_pattern(events, "Megalo Square25", start, pattern, 34, shift=-12)

        # Emotional harmonic bed: sparse piano hits, not constant arpeggio spam.
        add_chord(events, "Hopes Piano1", start, chords[bar], 0.55, 54)
        add_chord(events, "Hopes Piano1", start + 2.0, chords[bar], 0.48, 48)

        root = nn(roots[bar])
        fifth = root + 7
        octave = root + 12
        bass_line = [root, root, fifth, octave, root, fifth, octave, fifth]
        for step, pitch in enumerate(bass_line):
            vel = 94 if step in (0, 4) else 72
            add_note(events, "Megalo Bass", start + step * 0.5, pitch, 0.38, vel)

        # Low guitar rhythm leaves space for the melody.
        power = [root + 12, fifth + 12, octave + 12]
        for pos in (0.0, 1.5, 2.0, 3.5):
            add_chord(events, "Megalo Bg Guitar", start + pos, power, 0.27, 64)
        if bar >= 4:
            add_chord(events, "Megalo Over Guitar", start, power, 0.42, 70)
            add_chord(events, "Megalo Over Guitar", start + 2.0, power, 0.42, 66)

        # Straight powerful beat; fills only close the two four-bar phrases.
        add_note(events, "Hopes POWER", start, 36, 0.08, 108)
        add_note(events, "Hopes POWER", start + 2.0, 36, 0.08, 102)
        add_note(events, "Hopes POWER", start + 1.0, 38, 0.09, 112)
        add_note(events, "Hopes POWER", start + 3.0, 38, 0.09, 116)
        for step in range(8):
            add_note(events, "Hopes POWER", start + step * 0.5, 42, 0.05,
                     52 if step % 2 else 64)
        if bar in (3, 7):
            for offset, drum_note in ((3.0, 45), (3.25, 47), (3.5, 48), (3.75, 50)):
                add_note(events, "Hopes POWER", start + offset, drum_note, 0.06, 84)
        if bar in (0, 4):
            add_note(events, "Megalo Impact", start, roots[bar], 0.24, 92)

    # Final E-minor hit.
    end = BARS * 4.0 - 0.5
    final = ["E2", "B2", "E3", "G3", "B3"]
    add_chord(events, "Megalo Over Guitar", end, final, 0.55, 108)
    add_chord(events, "Hopes Piano1", end, final, 0.55, 106)
    add_note(events, "Hopes POWER", end, 49, 0.22, 120)

    return events


def render_stem(soundfont: Path, bank: int, preset: int,
                events: Sequence[Event], gain: float = 0.78) -> np.ndarray:
    total_frames = int((BARS * BAR_SECONDS + TAIL_SECONDS) * SR)
    synth = fluidsynth.Synth(gain=gain, samplerate=SR)
    sfid = synth.sfload(str(soundfont))
    result = synth.program_select(0, sfid, bank, preset)
    if result != 0:
        synth.delete()
        raise RuntimeError(f"Could not select exact bank={bank} preset={preset}")
    synth.set_reverb(roomsize=0.13, damping=0.72, width=0.68, level=0.07)
    synth.set_chorus(nr=2, level=0.10, speed=0.24, depth=1.6, type=0)

    timeline = []
    for start, duration, note, velocity in events:
        timeline.append((int(start * BEAT * SR), True, note, velocity))
        timeline.append((int((start + duration) * BEAT * SR), False, note, 0))
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


def soft_pan(audio: np.ndarray, pan: float) -> np.ndarray:
    out = audio.copy()
    if pan < 0:
        out[:, 1] *= 1.0 + pan * 0.55
    elif pan > 0:
        out[:, 0] *= 1.0 - pan * 0.55
    return out


def write_wav(path: Path, audio: np.ndarray) -> None:
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak * 0.94
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes(pcm.tobytes())


def encode(wav_path: Path, mp3_path: Path) -> None:
    filt = (
        "highpass=f=30,lowpass=f=11800,"
        "acompressor=threshold=-18dB:ratio=2.1:attack=10:release=120,"
        "alimiter=limit=0.96"
    )
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path),
        "-af", filt, "-codec:a", "libmp3lame", "-q:a", "2", str(mp3_path),
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
        bank, preset, _ = PRESETS[name]
        track = MidiTrack()
        track.append(MetaMessage("track_name", name=name, time=0))
        track.append(Message("control_change", channel=midi_channel, control=0,
                             value=min(127, bank), time=0))
        track.append(Message("program_change", channel=midi_channel,
                             program=min(127, preset), time=0))
        timeline = []
        for start, duration, note, velocity in track_events:
            timeline.append((round(start * TPB), True, note, velocity))
            timeline.append((round((start + duration) * TPB), False, note, 0))
        timeline.sort(key=lambda item: (item[0], item[1]))
        previous = 0
        for tick, is_on, note, velocity in timeline:
            track.append(Message(
                "note_on" if is_on else "note_off",
                channel=midi_channel,
                note=note,
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

    events = make_events()
    total = int((BARS * BAR_SECONDS + TAIL_SECONDS) * SR)
    melody_mix = np.zeros((total, 2), dtype=np.float64)
    context_mix = np.zeros((total, 2), dtype=np.float64)

    melody_gains = {
        "Hopes Lead Guitar": (0.92, -0.08),
        "Hopes Piano1": (0.32, 0.10),
        "Megalo Square25": (0.12, 0.0),
    }
    context_gains = {
        "Hopes Lead Guitar": (0.80, -0.08),
        "Hopes Piano1": (0.34, 0.10),
        "Megalo Square25": (0.14, 0.0),
        "Megalo Bass": (0.64, 0.0),
        "Megalo Bg Guitar": (0.40, 0.20),
        "Megalo Over Guitar": (0.52, -0.20),
        "Hopes POWER": (0.70, 0.0),
        "Megalo Impact": (0.34, 0.0),
    }

    instrument_lines = [
        f"BPM: {BPM}",
        "Key: E minor",
        "Completely new manually written eight-bar melody; no random generator.",
        "No saxophone, clarinet, flute, trumpet, brass, violin, strings, choir, or organ.",
        "Only MEGALOVANIA and Hopes and Dreams track-group presets are active.",
        "",
    ]

    for name, track_events in events.items():
        if not track_events:
            continue
        bank, preset, source = PRESETS[name]
        instrument_lines.append(f"{name}: bank={bank} preset={preset} | {source}")
        stem = render_stem(args.soundfont, bank, preset, track_events)
        length = min(total, len(stem))
        if name in melody_gains:
            gain, pan = melody_gains[name]
            melody_mix[:length] += soft_pan(stem[:length], pan) * gain
        if name in context_gains:
            gain, pan = context_gains[name]
            context_mix[:length] += soft_pan(stem[:length], pan) * gain

    melody_mix = np.tanh(melody_mix * 1.02)
    context_mix = np.tanh(context_mix * 1.05)

    melody_wav = args.out / "KRIS_NEW_MELODY_V3_MELODY.wav"
    context_wav = args.out / "KRIS_NEW_MELODY_V3_CONTEXT.wav"
    write_wav(melody_wav, melody_mix)
    write_wav(context_wav, context_mix)
    encode(melody_wav, args.out / "KRIS_NEW_MELODY_V3_MELODY.mp3")
    encode(context_wav, args.out / "KRIS_NEW_MELODY_V3_CONTEXT.mp3")
    export_midi(args.out / "KRIS_NEW_MELODY_V3.mid", events)
    (args.out / "KRIS_NEW_MELODY_V3_INSTRUMENTS.txt").write_text(
        "\n".join(instrument_lines) + "\n", encoding="utf-8"
    )
    print("\n".join(instrument_lines))


if __name__ == "__main__":
    main()
