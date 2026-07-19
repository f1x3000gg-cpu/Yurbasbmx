#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import wave
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import fluidsynth
import numpy as np
from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo

SR = 44100
BPM = 136
BEAT_SECONDS = 60.0 / BPM
BARS = 68
TAIL_SECONDS = 2.5
TPB = 480

PITCH_CLASS = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

Event = Tuple[float, float, int, int]
Pattern = Sequence[Tuple[str | None, float]]

PRESETS = {
    "MEGALOVANIA Lead Guitar": (0, 0, "100 MEGALOVANIA - Overdriven Guitar"),
    "MEGALOVANIA Background Guitar": (0, 13, "100 MEGALOVANIA - Background Guitar"),
    "MEGALOVANIA Bass": (0, 14, "100 MEGALOVANIA - Bass"),
    "ASGORE Piano": (2, 64, "077 ASGORE - Piano"),
    "ASGORE Timpani": (2, 74, "077 ASGORE - Timpani"),
    "ASGORE Drums": (2, 80, "077 ASGORE - Drumkit"),
}


def note_number(note: str) -> int:
    if len(note) >= 3 and note[1] in "#b":
        name, octave = note[:2], int(note[2:])
    else:
        name, octave = note[0], int(note[1:])
    return 12 * (octave + 1) + PITCH_CLASS[name]


def add(events: Dict[str, List[Event]], track: str, start: float,
        note: str | int, duration: float, velocity: int) -> None:
    midi_note = note if isinstance(note, int) else note_number(note)
    events[track].append((float(start), float(duration), int(midi_note), int(velocity)))


def add_pattern(events: Dict[str, List[Event]], track: str, start: float,
                notes: Pattern, velocity: int, gate: float = 0.86) -> None:
    cursor = start
    for note, duration in notes:
        if note is not None:
            add(events, track, cursor, note, max(0.05, duration * gate), velocity)
        cursor += duration


def add_chord(events: Dict[str, List[Event]], track: str, start: float,
              notes: Iterable[str], duration: float, velocity: int) -> None:
    for note in notes:
        add(events, track, start, note, duration, velocity)


def transpose_pattern(pattern: Pattern, semitones: int) -> List[Tuple[int | None, float]]:
    result: List[Tuple[int | None, float]] = []
    for note, duration in pattern:
        result.append((None if note is None else note_number(note) + semitones, duration))
    return result


THEME_A: List[Pattern] = [
    [("E4", .5), ("A4", .5), ("B4", .5), ("C5", .5), ("B4", .5), ("A4", .5), ("E4", 1.0)],
    [("G4", .5), ("A4", .5), ("C5", 1.0), ("B4", .5), ("A4", .5), ("G4", 1.0)],
    [("F4", .5), ("A4", .5), ("C5", .5), ("D5", .5), ("C5", .5), ("A4", .5), ("F4", 1.0)],
    [("E4", .5), ("G#4", .5), ("B4", 1.0), ("A4", .5), ("G#4", .5), ("E4", 1.0)],
    [("A4", .5), ("C5", .5), ("E5", 1.0), ("D5", .5), ("C5", .5), ("B4", 1.0)],
    [("G4", .5), ("B4", .5), ("D5", 1.0), ("C5", .5), ("B4", .5), ("A4", 1.0)],
    [("F4", .5), ("A4", .5), ("C5", 1.0), ("B4", .5), ("A4", .5), ("G4", 1.0)],
    [("E4", .5), ("G#4", .5), ("B4", .5), ("D5", .5), ("C5", .5), ("B4", .5), ("A4", 1.0)],
]

THEME_B: List[Pattern] = [
    [("C5", .75), ("B4", .25), ("A4", .5), ("E4", .5), ("G4", .5), ("A4", .5), ("C5", 1.0)],
    [("B4", .5), ("G4", .5), ("E4", 1.0), ("F4", .5), ("G4", .5), ("A4", 1.0)],
    [("A4", .5), ("C5", .5), ("E5", .75), ("D5", .25), ("C5", .5), ("B4", .5), ("A4", 1.0)],
    [("G4", .5), ("A4", .5), ("C5", 1.0), ("B4", .5), ("G4", .5), ("E4", 1.0)],
    [("E4", .5), ("G4", .5), ("A4", 1.0), ("C5", .5), ("B4", .5), ("A4", 1.0)],
    [("D5", .5), ("C5", .5), ("B4", .5), ("A4", .5), ("G4", .5), ("A4", .5), ("B4", 1.0)],
    [("C5", .5), ("A4", .5), ("F4", 1.0), ("G4", .5), ("A4", .5), ("C5", 1.0)],
    [("B4", .5), ("G#4", .5), ("E4", 1.0), ("G#4", .5), ("B4", .5), ("A4", 1.0)],
]

CHORUS: List[Pattern] = [
    [("A4", .5), ("C5", .5), ("E5", .5), ("A5", .5), ("G5", .5), ("E5", .5), ("C5", 1.0)],
    [("G4", .5), ("C5", .5), ("E5", 1.0), ("D5", .5), ("C5", .5), ("G4", 1.0)],
    [("B4", .5), ("D5", .5), ("G5", .75), ("F5", .25), ("E5", .5), ("D5", .5), ("B4", 1.0)],
    [("A4", .5), ("C5", .5), ("F5", 1.0), ("E5", .5), ("C5", .5), ("A4", 1.0)],
    [("C5", .5), ("E5", .5), ("A5", 1.0), ("G5", .5), ("E5", .5), ("D5", 1.0)],
    [("B4", .5), ("D5", .5), ("G5", 1.0), ("E5", .5), ("D5", .5), ("C5", 1.0)],
    [("A4", .5), ("D5", .5), ("F5", 1.0), ("E5", .5), ("D5", .5), ("C5", 1.0)],
    [("B4", .5), ("E5", .5), ("G#5", .5), ("B5", .5), ("A5", .5), ("E5", .5), ("A4", 1.0)],
]

BRIDGE: List[Pattern] = [
    [("F4", 1.0), ("A4", .5), ("C5", .5), ("D5", 1.0), ("C5", 1.0)],
    [("E4", .5), ("A4", .5), ("C5", 1.0), ("B4", 1.0), ("A4", 1.0)],
    [("F4", .5), ("G4", .5), ("A4", 1.0), ("C5", .5), ("D5", .5), ("E5", 1.0)],
    [("G4", .5), ("E4", .5), ("C4", 1.0), ("E4", .5), ("G4", .5), ("A4", 1.0)],
    [("D5", .5), ("C5", .5), ("A4", 1.0), ("F4", .5), ("A4", .5), ("C5", 1.0)],
    [("E5", .5), ("C5", .5), ("A4", 1.0), ("B4", .5), ("C5", .5), ("E5", 1.0)],
    [("F5", .5), ("E5", .5), ("D5", 1.0), ("C5", .5), ("B4", .5), ("A4", 1.0)],
    [("G#4", .5), ("B4", .5), ("E5", 1.0), ("D5", .5), ("B4", .5), ("A4", 1.0)],
]

CHORDS = {
    "Am": (["A3", "C4", "E4"], "A2"),
    "F": (["F3", "A3", "C4"], "F2"),
    "C": (["C3", "E3", "G3"], "C3"),
    "G": (["G3", "B3", "D4"], "G2"),
    "Dm": (["D3", "F3", "A3"], "D3"),
    "E": (["E3", "G#3", "B3"], "E2"),
}

INTRO_PROG = ["Am", "F", "C", "G"]
THEME_A_PROG = ["Am", "F", "C", "G", "Am", "F", "Dm", "E"]
THEME_B_PROG = ["C", "G", "Am", "F", "C", "G", "F", "E"]
CHORUS_PROG = ["Am", "C", "G", "F", "Am", "C", "Dm", "E"]
INTERLUDE_PROG = ["F", "G", "Am", "E"]
BRIDGE_PROG = ["Dm", "Am", "F", "C", "Dm", "Am", "F", "E"]
OUTRO_PROG = ["Am", "F", "E", "Am"]


def section_progression(bar: int) -> str:
    if bar < 4:
        return INTRO_PROG[bar]
    if bar < 12:
        return THEME_A_PROG[bar - 4]
    if bar < 20:
        return THEME_B_PROG[bar - 12]
    if bar < 28:
        return CHORUS_PROG[bar - 20]
    if bar < 32:
        return INTERLUDE_PROG[bar - 28]
    if bar < 40:
        return THEME_A_PROG[bar - 32]
    if bar < 48:
        return BRIDGE_PROG[bar - 40]
    if bar < 64:
        return CHORUS_PROG[(bar - 48) % 8]
    return OUTRO_PROG[bar - 64]


def add_harmony(events: Dict[str, List[Event]], bar: int, chord_name: str) -> None:
    start = bar * 4.0
    chord_notes, root = CHORDS[chord_name]

    if bar < 4:
        piano_velocity = 42
    elif 28 <= bar < 32:
        piano_velocity = 62
    elif 40 <= bar < 48:
        piano_velocity = 55
    else:
        piano_velocity = 48
    add_chord(events, "ASGORE Piano", start, chord_notes, 1.35, piano_velocity)
    add_chord(events, "ASGORE Piano", start + 2.0, chord_notes, 1.20, piano_velocity - 6)

    root_midi = note_number(root)
    fifth_midi = root_midi + 7
    octave_midi = root_midi + 12
    bg_velocity = 38 if bar < 20 else 44
    for pos in (0.0, 2.0):
        add(events, "MEGALOVANIA Background Guitar", start + pos,
            octave_midi, 1.45, bg_velocity)
        add(events, "MEGALOVANIA Background Guitar", start + pos,
            fifth_midi + 12, 1.45, bg_velocity - 5)

    bass_root = root_midi
    if bass_root < note_number("C2"):
        bass_root += 12
    bass_pattern = [(0.0, bass_root), (1.5, bass_root + 7),
                    (2.0, bass_root + 12), (3.0, bass_root + 7)]
    bass_velocity = 56 if bar < 20 else 62
    for pos, pitch in bass_pattern:
        add(events, "MEGALOVANIA Bass", start + pos, pitch, .42, bass_velocity)


def add_drums(events: Dict[str, List[Event]], bar: int) -> None:
    start = bar * 4.0
    for pos in (0.0, 2.0):
        add(events, "ASGORE Drums", start + pos, 36, .07, 92)
    if bar >= 20:
        add(events, "ASGORE Drums", start + 2.5, 36, .06, 66)
    for pos in (1.0, 3.0):
        add(events, "ASGORE Drums", start + pos, 38, .08, 104)
    for step in range(8):
        velocity = 42 if step % 2 else 51
        if 28 <= bar < 32:
            velocity -= 10
        add(events, "ASGORE Drums", start + step * .5, 42, .04, velocity)

    if bar in (11, 19, 27, 39, 47, 55, 63, 66):
        for off, drum, vel in ((3.0, 45, 68), (3.25, 47, 72),
                               (3.5, 48, 78), (3.75, 50, 84)):
            add(events, "ASGORE Drums", start + off, drum, .05, vel)
    if bar in (19, 27, 39, 47, 55, 63):
        chord_name = section_progression(bar)
        _, root = CHORDS[chord_name]
        add(events, "ASGORE Timpani", start + 3.0,
            note_number(root) + 12, .75, 72)


def add_main_melody(events: Dict[str, List[Event]]) -> None:
    add_pattern(events, "MEGALOVANIA Lead Guitar", 2 * 4.0,
                [("E4", .5), ("A4", .5), ("B4", .5), ("C5", .5),
                 ("B4", .5), ("A4", .5), ("E4", 1.0)], 67, .88)
    add_pattern(events, "MEGALOVANIA Lead Guitar", 3 * 4.0,
                [("G4", .5), ("A4", .5), ("C5", 1.0),
                 ("B4", .5), ("G#4", .5), ("A4", 1.0)], 69, .88)

    for index, pat in enumerate(THEME_A):
        add_pattern(events, "MEGALOVANIA Lead Guitar", (4 + index) * 4.0, pat, 72, .88)
    for index, pat in enumerate(THEME_B):
        add_pattern(events, "MEGALOVANIA Lead Guitar", (12 + index) * 4.0, pat, 70, .90)
    for index, pat in enumerate(CHORUS):
        add_pattern(events, "MEGALOVANIA Lead Guitar", (20 + index) * 4.0, pat, 75, .90)

    interlude = [
        [("A4", .5), ("C5", .5), ("A4", .5), ("F4", .5), ("G4", 1.0), ("A4", 1.0)],
        [("B4", .5), ("D5", .5), ("B4", .5), ("G4", .5), ("A4", 1.0), ("B4", 1.0)],
        [("C5", .5), ("E5", .5), ("C5", .5), ("A4", .5), ("B4", 1.0), ("C5", 1.0)],
        [("B4", .5), ("G#4", .5), ("E4", 1.0), ("G#4", .5), ("B4", .5), ("A4", 1.0)],
    ]
    for index, pat in enumerate(interlude):
        add_pattern(events, "ASGORE Piano", (28 + index) * 4.0, pat, 76, .82)
        if index in (1, 3):
            add_pattern(events, "MEGALOVANIA Lead Guitar", (28 + index) * 4.0 + 2.0,
                        [("E4", .5), ("G4", .5), ("A4", 1.0), (None, 2.0)],
                        58, .72)

    for index, pat in enumerate(THEME_A):
        varied: List[Tuple[int | None, float]] = []
        for note, duration in pat:
            if note is None:
                varied.append((None, duration))
            else:
                pitch = note_number(note)
                varied.append((pitch + (12 if index >= 4 and pitch < note_number("C5") else 0), duration))
        add_pattern(events, "MEGALOVANIA Lead Guitar", (32 + index) * 4.0,
                    varied, 71, .84)

    for index, pat in enumerate(BRIDGE):
        add_pattern(events, "MEGALOVANIA Lead Guitar", (40 + index) * 4.0, pat, 68, .92)

    for index, pat in enumerate(CHORUS):
        start = (48 + index) * 4.0
        add_pattern(events, "MEGALOVANIA Lead Guitar", start, pat, 76, .90)
        answer = transpose_pattern(pat, -12)
        add_pattern(events, "ASGORE Piano", start + .25, answer, 44, .76)

    for index, pat in enumerate(CHORUS):
        start = (56 + index) * 4.0
        add_pattern(events, "MEGALOVANIA Lead Guitar", start, pat, 78, .92)
        if index in (2, 3, 6, 7):
            compact = [(note, duration) for note, duration in pat[:4]]
            add_pattern(events, "ASGORE Piano", start + 2.0,
                        transpose_pattern(compact, -12), 48, .70)

    outro = [
        [("E5", .5), ("C5", .5), ("A4", 1.0), ("G4", .5), ("A4", .5), ("C5", 1.0)],
        [("C5", .5), ("A4", .5), ("F4", 1.0), ("A4", .5), ("C5", .5), ("E5", 1.0)],
        [("B4", .5), ("G#4", .5), ("E4", 1.0), ("G#4", .5), ("B4", .5), ("A4", 1.0)],
        [("A4", 4.0)],
    ]
    for index, pat in enumerate(outro):
        add_pattern(events, "MEGALOVANIA Lead Guitar", (64 + index) * 4.0,
                    pat, 66 if index < 3 else 60, .94)


def build_events() -> Dict[str, List[Event]]:
    events: Dict[str, List[Event]] = {name: [] for name in PRESETS}
    for bar in range(BARS):
        chord_name = section_progression(bar)
        add_harmony(events, bar, chord_name)
        add_drums(events, bar)
    add_main_melody(events)
    return events


def render_stem(soundfont: Path, bank: int, preset: int,
                events: Sequence[Event]) -> np.ndarray:
    total_frames = int(((BARS * 4 * BEAT_SECONDS) + TAIL_SECONDS) * SR)
    synth = fluidsynth.Synth(gain=.64, samplerate=SR)
    sfid = synth.sfload(str(soundfont))
    if synth.program_select(0, sfid, bank, preset) != 0:
        synth.delete()
        raise RuntimeError(f"Cannot select bank={bank} preset={preset}")
    synth.set_reverb(roomsize=.12, damping=.74, width=.70, level=.055)
    synth.set_chorus(nr=2, level=.035, speed=.22, depth=.80, type=0)

    timeline: List[Tuple[int, int, int, int]] = []
    for start, duration, note, velocity in events:
        timeline.append((int(start * BEAT_SECONDS * SR), 1, note, velocity))
        timeline.append((int((start + duration) * BEAT_SECONDS * SR), 0, note, 0))
    timeline.sort(key=lambda item: (item[0], item[1]))

    chunks: List[np.ndarray] = []
    cursor = 0
    for frame, on, note, velocity in timeline:
        if frame > cursor:
            chunks.append(synth.get_samples(frame - cursor))
            cursor = frame
        if on:
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


def mix_tracks(soundfont: Path, events: Dict[str, List[Event]]) -> np.ndarray:
    total_frames = int(((BARS * 4 * BEAT_SECONDS) + TAIL_SECONDS) * SR)
    output = np.zeros((total_frames, 2), dtype=np.float64)
    gains = {
        "MEGALOVANIA Lead Guitar": .50,
        "MEGALOVANIA Background Guitar": .20,
        "MEGALOVANIA Bass": .29,
        "ASGORE Piano": .32,
        "ASGORE Timpani": .30,
        "ASGORE Drums": .53,
    }
    pans = {
        "MEGALOVANIA Lead Guitar": 0.0,
        "MEGALOVANIA Background Guitar": -0.20,
        "MEGALOVANIA Bass": 0.0,
        "ASGORE Piano": 0.18,
        "ASGORE Timpani": 0.0,
        "ASGORE Drums": 0.0,
    }

    for name, track_events in events.items():
        bank, preset, _ = PRESETS[name]
        stem = render_stem(soundfont, bank, preset, track_events)
        pan = pans[name]
        left = np.sqrt((1.0 - pan) * .5)
        right = np.sqrt((1.0 + pan) * .5)
        stem[:, 0] *= left
        stem[:, 1] *= right
        output[:len(stem)] += stem * gains[name]

    peak = float(np.max(np.abs(output)))
    if peak > 0:
        output = output / peak * .91
    return output


def write_wav(path: Path, audio: np.ndarray) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes(pcm.tobytes())


def encode_audio(wav_path: Path, mp3_path: Path, ogg_path: Path) -> None:
    common_filter = (
        "highpass=f=48,lowpass=f=14500,"
        "acompressor=threshold=-17dB:ratio=1.55:attack=14:release=150,"
        "alimiter=limit=.95"
    )
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path),
        "-af", common_filter, "-codec:a", "libmp3lame", "-q:a", "2",
        str(mp3_path),
    ], check=True)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path),
        "-af", common_filter, "-codec:a", "libvorbis", "-q:a", "6",
        str(ogg_path),
    ], check=True)


def export_midi(path: Path, events: Dict[str, List[Event]]) -> None:
    midi = MidiFile(ticks_per_beat=TPB)
    tempo = MidiTrack()
    tempo.append(MetaMessage("set_tempo", tempo=bpm2tempo(BPM), time=0))
    tempo.append(MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    midi.tracks.append(tempo)

    channel = 0
    for name, track_events in events.items():
        while channel == 9:
            channel += 1
        ch = channel % 16
        channel += 1
        bank, preset, _ = PRESETS[name]
        track = MidiTrack()
        track.append(MetaMessage("track_name", name=name, time=0))
        track.append(Message("control_change", channel=ch, control=0, value=bank, time=0))
        track.append(Message("program_change", channel=ch, program=preset, time=0))
        timeline: List[Tuple[int, int, int, int]] = []
        for start, duration, note, velocity in track_events:
            timeline.append((round(start * TPB), 1, note, velocity))
            timeline.append((round((start + duration) * TPB), 0, note, 0))
        timeline.sort(key=lambda item: (item[0], item[1]))
        previous = 0
        for tick, on, note, velocity in timeline:
            track.append(Message(
                "note_on" if on else "note_off",
                channel=ch,
                note=note,
                velocity=velocity if on else 0,
                time=tick - previous,
            ))
            previous = tick
        midi.tracks.append(track)
    midi.save(path)


def write_instrument_list(path: Path) -> None:
    duration = BARS * 4 * BEAT_SECONDS
    lines = [
        "KRIS FULL TWO-MINUTE BATTLE THEME",
        f"Tempo: {BPM} BPM",
        f"Musical duration before tail: {duration:.2f} seconds",
        f"Structure: {BARS} bars / intro / theme A / theme B / chorus / interlude / variation / bridge / two final choruses / outro",
        "Melody: fully hand-written; no random generator; no Undertale melody copied",
        "",
        "Presets:",
    ]
    for _, (_, _, label) in PRESETS.items():
        lines.append(f"- {label}")
    lines += [
        "",
        "Lead approach:",
        "- MEGALOVANIA Overdriven Guitar keeps its original preset tone.",
        "- Notes stay mainly in the middle register; velocity and gain are moderate.",
        "- No octave-down heavy doubling and no octave-up screaming doubling.",
        "- ASGORE Piano is harmony/countermelody, not a fake replacement lead.",
        "- Approved ASGORE drum groove is retained and developed with section fills.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--soundfont", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    events = build_events()
    audio = mix_tracks(args.soundfont, events)

    wav_path = args.out / "KRIS_FULL_TWO_MINUTE_THEME.wav"
    mp3_path = args.out / "KRIS_FULL_TWO_MINUTE_THEME.mp3"
    ogg_path = args.out / "KRIS_FULL_TWO_MINUTE_THEME_GAME.ogg"
    midi_path = args.out / "KRIS_FULL_TWO_MINUTE_THEME.mid"
    info_path = args.out / "KRIS_FULL_TWO_MINUTE_THEME_INSTRUMENTS.txt"

    write_wav(wav_path, audio)
    encode_audio(wav_path, mp3_path, ogg_path)
    export_midi(midi_path, events)
    write_instrument_list(info_path)

    print(f"Rendered {mp3_path}")
    print(f"Duration target: {BARS * 4 * BEAT_SECONDS:.2f}s + tail")


if __name__ == "__main__":
    main()
