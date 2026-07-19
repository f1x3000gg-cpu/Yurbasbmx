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
BPM = 154
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
Pattern = Sequence[Tuple[str | int | None, float]]

# Only exact presets from the MEGALOVANIA and Hopes and Dreams groups in
# UNDERTALE Soundfont 2. No saxophone, trumpet, clarinet, flute, brass,
# violin, strings, or generic GM fallback.
PRESETS = {
    "Hopes Lead Guitar": (3, 8, "087 Hopes and Dreams - Lead Guitar"),
    "Hopes Piano 1":     (2, 126, "087 Hopes and Dreams - Piano 1"),
    "Hopes POWER":       (2, 127, "087 Hopes and Dreams - POWER DrumKit"),
    "Hopes Pulse 25":    (3, 5, "087 Hopes and Dreams - Pulse 25%"),
    "Megalo Over Guitar":(0, 0, "100 MEGALOVANIA - Overdriven Guitar"),
    "Megalo Square 25":  (0, 8, "100 MEGALOVANIA - 25% Square"),
    "Megalo Rock Organ": (0, 12, "100 MEGALOVANIA - Rock Organ"),
    "Megalo Bg Guitar":  (0, 13, "100 MEGALOVANIA - Background Guitar"),
    "Megalo Bass":       (0, 14, "100 MEGALOVANIA - Bass"),
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
    events[track].append((start, duration, nn(note), velocity))


def add_pattern(events: Dict[str, List[Event]], track: str, start: float,
                pattern: Pattern, velocity: int, shift: int = 0) -> None:
    cursor = start
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

    # G minor. One deliberately written eight-bar song phrase:
    # bars 1-2 state the motif, 3-4 answer it, 5-6 open into the refrain,
    # and 7-8 resolve it. No random-note generation and no fast lead spam.
    melody: List[Pattern] = [
        [(None, .5), ("D5", .5), ("G5", 1), ("F5", .5), ("D5", .5), ("C5", 1)],
        [("Bb4", .5), ("D5", .5), ("F5", 1), ("Eb5", .5), ("D5", .5), ("C5", 1)],
        [("G4", .5), ("Bb4", .5), ("D5", 1), ("C5", .5), ("Bb4", .5), ("G4", 1)],
        [("A4", .5), ("Bb4", .5), ("C5", 1), ("D5", .5), ("F#4", .5), ("G4", 1)],
        [("D5", .5), ("F5", .5), ("G5", 1), ("Bb5", .5), ("A5", .5), ("G5", 1)],
        [("F5", .5), ("D5", .5), ("Eb5", 1), ("C5", .5), ("Bb4", .5), ("A4", 1)],
        [("Bb4", .5), ("D5", .5), ("F5", .5), ("G5", .5), ("F5", 1), ("D5", 1)],
        [("C5", .5), ("Bb4", .5), ("A4", 1), ("F#4", .5), ("A4", .5), ("G4", 1)],
    ]

    chords = [
        ["G2", "D3", "G3", "Bb3"],
        ["Eb2", "Bb2", "Eb3", "G3"],
        ["Bb2", "F3", "Bb3", "D4"],
        ["D2", "A2", "D3", "F#3"],
        ["G2", "D3", "G3", "Bb3"],
        ["Eb2", "Bb2", "Eb3", "G3"],
        ["C2", "G2", "C3", "Eb3"],
        ["D2", "A2", "D3", "F#3"],
    ]
    roots = ["G1", "Eb1", "Bb1", "D2", "G1", "Eb1", "C2", "D2"]

    for bar in range(BARS):
        start = bar * 4.0
        pattern = melody[bar]

        # Lead remains one clear singable line. The organ only adds weight one
        # octave below, and the square is quiet enough not to become squeaky.
        add_pattern(events, "Hopes Lead Guitar", start, pattern, 108 if bar >= 4 else 100)
        add_pattern(events, "Megalo Rock Organ", start, pattern, 60, shift=-12)
        add_pattern(events, "Megalo Square 25", start, pattern, 28, shift=-12)

        # Piano punctuation rather than a piano-led arrangement.
        add_chord(events, "Hopes Piano 1", start, chords[bar], 0.42, 54)
        add_chord(events, "Hopes Piano 1", start + 2.0, chords[bar], 0.36, 48)

        # Sustained pulse gives the Asriel-like emotional bed without strings.
        add_note(events, "Hopes Pulse 25", start, chords[bar][1], 1.75, 34)
        add_note(events, "Hopes Pulse 25", start + 2.0, chords[bar][2], 1.75, 32)

        # Bass is intentionally simple: root, fifth, root, octave.
        root = nn(roots[bar])
        bass_sequence = [root, root + 7, root, root + 12]
        for beat_index, pitch in enumerate(bass_sequence):
            add_note(events, "Megalo Bass", start + beat_index, pitch, 0.72, 88 if beat_index == 0 else 72)

        # Guitar rhythm grows only in the refrain. It never doubles every lead note.
        power_root = nn(chords[bar][0]) + 12
        power = [power_root, power_root + 7, power_root + 12]
        add_chord(events, "Megalo Over Guitar", start, power, 0.40, 76)
        add_chord(events, "Megalo Over Guitar", start + 2.0, power, 0.36, 70)
        chug_positions = [0.0, 1.5, 2.0, 3.5] if bar < 4 else [0.0, .75, 1.5, 2.0, 2.75, 3.5]
        for pos in chug_positions:
            add_chord(events, "Megalo Bg Guitar", start + pos, power, 0.20, 50 if bar < 4 else 58)

        # POWER drums: steady groove, not breakcore spam.
        for pos in (0.0, 2.0):
            add_note(events, "Hopes POWER", start + pos, 36, 0.08, 104)
        if bar >= 4:
            add_note(events, "Hopes POWER", start + 2.5, 36, 0.07, 78)
        for pos in (1.0, 3.0):
            add_note(events, "Hopes POWER", start + pos, 38, 0.09, 112)
        for step in range(8):
            add_note(events, "Hopes POWER", start + step * .5, 42, 0.05, 42 if step % 2 else 55)
        if bar in (3, 7):
            for pos, drum in ((3.0, 45), (3.25, 47), (3.5, 48), (3.75, 50)):
                add_note(events, "Hopes POWER", start + pos, drum, 0.06, 76 if bar == 3 else 88)

    return events


def render_stem(soundfont: Path, bank: int, preset: int,
                events: Sequence[Event]) -> np.ndarray:
    total_frames = int((BARS * 4 * BEAT + TAIL) * SR)
    synth = fluidsynth.Synth(gain=0.78, samplerate=SR)
    sfid = synth.sfload(str(soundfont))
    result = synth.program_select(0, sfid, bank, preset)
    if result != 0:
        synth.delete()
        raise RuntimeError(f"Failed exact preset bank={bank} preset={preset}")
    synth.set_reverb(roomsize=.12, damping=.7, width=.65, level=.07)
    synth.set_chorus(nr=2, level=.10, speed=.24, depth=1.6, type=0)

    timeline: List[Tuple[int, bool, int, int]] = []
    for start, duration, note, velocity in events:
        timeline.append((int(start * BEAT * SR), True, note, velocity))
        timeline.append((int((start + duration) * BEAT * SR), False, note, 0))
    timeline.sort(key=lambda item: (item[0], item[1]))

    chunks: List[np.ndarray] = []
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

    raw = np.concatenate(chunks).astype(np.float64) / 32768.0
    return raw.reshape(-1, 2)


def write_wav(path: Path, audio: np.ndarray) -> None:
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak * .94
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes(pcm.tobytes())


def encode(wav_path: Path, mp3_path: Path) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path),
        "-af", "highpass=f=30,lowpass=f=12000,acompressor=threshold=-18dB:ratio=2:attack=10:release=120,alimiter=limit=0.96",
        "-codec:a", "libmp3lame", "-q:a", "2", str(mp3_path),
    ], check=True)


def export_midi(path: Path, events: Dict[str, List[Event]]) -> None:
    midi = MidiFile(ticks_per_beat=TPB)
    tempo_track = MidiTrack()
    tempo_track.append(MetaMessage("track_name", name="Tempo", time=0))
    tempo_track.append(MetaMessage("set_tempo", tempo=bpm2tempo(BPM), time=0))
    tempo_track.append(MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    midi.tracks.append(tempo_track)

    channel = 0
    for name, track_events in events.items():
        if not track_events:
            continue
        if channel == 9:
            channel += 1
        midi_channel = channel % 16
        channel += 1
        bank, preset, _ = PRESETS[name]
        track = MidiTrack()
        track.append(MetaMessage("track_name", name=name, time=0))
        track.append(Message("control_change", channel=midi_channel, control=0, value=bank, time=0))
        track.append(Message("program_change", channel=midi_channel, program=preset, time=0))
        timeline: List[Tuple[int, bool, int, int]] = []
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
        midi.tracks.append(track)
    midi.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--soundfont", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    events = make_events()
    total_frames = int((BARS * 4 * BEAT + TAIL) * SR)
    context_mix = np.zeros((total_frames, 2), dtype=np.float64)
    melody_mix = np.zeros_like(context_mix)

    context_gains = {
        "Hopes Lead Guitar": .88,
        "Hopes Piano 1": .40,
        "Hopes POWER": .67,
        "Hopes Pulse 25": .25,
        "Megalo Over Guitar": .50,
        "Megalo Square 25": .19,
        "Megalo Rock Organ": .40,
        "Megalo Bg Guitar": .33,
        "Megalo Bass": .61,
    }
    melody_gains = {
        "Hopes Lead Guitar": .94,
        "Megalo Rock Organ": .40,
        "Megalo Square 25": .16,
    }

    instrument_lines = [
        f"BPM: {BPM}",
        "Key: G minor",
        "Eight-bar original melody proof; no random-note generator.",
        "Excluded completely: saxophone, trumpet, clarinet, flute, brass, violin, strings.",
        "Only MEGALOVANIA and Hopes and Dreams track-group presets are active.",
        "",
    ]

    for name, track_events in events.items():
        if not track_events:
            continue
        bank, preset, source = PRESETS[name]
        instrument_lines.append(f"{name}: bank={bank} preset={preset} | {source}")
        stem = render_stem(args.soundfont, bank, preset, track_events)
        length = min(total_frames, len(stem))
        context_mix[:length] += stem[:length] * context_gains[name]
        if name in melody_gains:
            melody_mix[:length] += stem[:length] * melody_gains[name]

    context_mix = np.tanh(context_mix * 1.04)
    melody_mix = np.tanh(melody_mix * 1.01)

    context_wav = args.out / "KRIS_ORGAN_GUITAR_8BAR_CONTEXT.wav"
    melody_wav = args.out / "KRIS_ORGAN_GUITAR_8BAR_MELODY.wav"
    write_wav(context_wav, context_mix)
    write_wav(melody_wav, melody_mix)
    encode(context_wav, args.out / "KRIS_ORGAN_GUITAR_8BAR_CONTEXT.mp3")
    encode(melody_wav, args.out / "KRIS_ORGAN_GUITAR_8BAR_MELODY.mp3")
    export_midi(args.out / "KRIS_ORGAN_GUITAR_8BAR.mid", events)
    (args.out / "KRIS_ORGAN_GUITAR_8BAR_INSTRUMENTS.txt").write_text(
        "\n".join(instrument_lines) + "\n", encoding="utf-8"
    )
    print("\n".join(instrument_lines))


if __name__ == "__main__":
    main()
