#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import subprocess
import wave
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import fluidsynth
import numpy as np
from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo

SR = 44100
BPM = 172
BEAT = 60.0 / BPM
BAR = 4.0 * BEAT
BARS = 40
TAIL = 2.4
TPB = 480

PITCH = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

Event = Tuple[float, float, int, int]
Pattern = Sequence[Tuple[str | int | None, float]]

# Exact preset locations inside UNDERTALE Soundfont 2.
# The final field is the source track represented by that preset group.
PRESETS = {
    # 087 Hopes and Dreams
    "Hopes Violin Detache": (2, 125, "087 Hopes and Dreams - Violin Detache"),
    "Hopes Piano 1":        (2, 126, "087 Hopes and Dreams - Piano 1"),
    "Hopes POWER":          (2, 127, "087 Hopes and Dreams - POWER DrumKit"),
    "Hopes Glockenspiel":   (3,   0, "087 Hopes and Dreams - Glockenspiel"),
    "Hopes Flute":          (3,   1, "087 Hopes and Dreams - Flute"),
    "Hopes Strings":        (3,   2, "087 Hopes and Dreams - Strings"),
    "Hopes Choir":          (3,   3, "087 Hopes and Dreams - Choir Aahs"),
    "Hopes Viola":          (3,   4, "087 Hopes and Dreams - Viola"),
    "Hopes Pulse 25":       (3,   5, "087 Hopes and Dreams - Pulse 25%"),
    "Hopes Pulse 50":       (3,   6, "087 Hopes and Dreams - Pulse 50%"),
    "Hopes Duty":           (3,   7, "087 Hopes and Dreams - Duty Cycle"),
    "Hopes Lead Guitar":    (3,   8, "087 Hopes and Dreams - Lead Guitar"),

    # 098 Battle Against a True Hero
    "Hero Square 25":       (3,  62, "098 Battle Against a True Hero - Square Wave 25%"),
    "Hero Duty":            (3,  63, "098 Battle Against a True Hero - Duty Cycle"),
    "Hero Piano":           (3,  64, "098 Battle Against a True Hero - Piano"),
    "Hero Romantic Tp":     (3,  65, "098 Battle Against a True Hero - Romantic Trumpet"),
    "Hero POWER":           (3,  66, "098 Battle Against a True Hero - POWER DrumKit"),
    "Hero Piano 1":         (3,  67, "098 Battle Against a True Hero - Piano 1"),
    "Hero Choir":           (3,  68, "098 Battle Against a True Hero - Choir Aahs"),
    "Hero Piccolo":         (3,  69, "098 Battle Against a True Hero - Piccolo"),
    "Hero Violin":          (3,  70, "098 Battle Against a True Hero - Violin"),
    "Hero Tubular Bells":   (3,  71, "098 Battle Against a True Hero - Tubular Bells"),
    "Hero Strings":         (3,  72, "098 Battle Against a True Hero - Strings"),
    "Hero Nylon Guitar":    (3,  73, "098 Battle Against a True Hero - Nylon-str. Guitar"),

    # 100 MEGALOVANIA
    "Megalo Over Guitar":   (0,   0, "100 MEGALOVANIA - Overdriven Guitar"),
    "Megalo Square":        (0,   1, "100 MEGALOVANIA - Square"),
    "Megalo Strings":       (0,   2, "100 MEGALOVANIA - Strings"),
    "Megalo Brass":         (0,   5, "100 MEGALOVANIA - Brass"),
    "Megalo Saw":           (0,   6, "100 MEGALOVANIA - Saw Wave"),
    "Megalo Violin":        (0,   7, "100 MEGALOVANIA - Violin"),
    "Megalo Square 25":     (0,   8, "100 MEGALOVANIA - 25% Square"),
    "Megalo Impact":        (0,   9, "100 MEGALOVANIA - Impact Hit"),
    "Megalo Ooh Choir":     (0,  10, "100 MEGALOVANIA - Ooh Choir"),
    "Megalo Organ 3":       (0,  11, "100 MEGALOVANIA - Organ 3"),
    "Megalo Rock Organ":    (0,  12, "100 MEGALOVANIA - Rock Organ"),
    "Megalo Bg Guitar":     (0,  13, "100 MEGALOVANIA - Background Guitar"),
    "Megalo Bass":          (0,  14, "100 MEGALOVANIA - Bass"),
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
            add_note(events, track, cursor, nn(note) + shift, duration * .94, velocity)
        cursor += duration


def add_chord(events: Dict[str, List[Event]], track: str, start: float,
              notes: Sequence[str | int], duration: float, velocity: int) -> None:
    for note in notes:
        add_note(events, track, start, note, duration, velocity)


def make_events() -> Dict[str, List[Event]]:
    events = {name: [] for name in PRESETS}

    # B minor. The melody was written as two complete eight-bar phrases before
    # any drums or orchestration were added. No random generator is used.
    hook: List[Pattern] = [
        [(None,.5),("F#4",.5),("B4",.75),("A4",.25),("D5",.5),("C#5",.5),("B4",1)],
        [("A4",.5),("F#4",.5),("E4",.5),("F#4",.5),("B4",1),("A4",.5),("F#4",.5)],
        [(None,.5),("D5",.5),("F#5",.75),("E5",.25),("C#5",.5),("B4",.5),("A4",1)],
        [("F#4",.5),("A4",.5),("B4",.5),("D5",.5),("C#5",.5),("A#4",.5),("B4",1)],
        [("B4",.5),("E5",.5),("G5",1),("F#5",.5),("E5",.5),("D5",1)],
        [("C#5",.5),("B4",.5),("G4",1),("A4",.5),("C#5",.5),("B4",1)],
        [("D5",1),("C#5",.5),("B4",.5),("A4",1),("F#4",1)],
        [("G4",.5),("A4",.5),("B4",1),("C#5",.5),("A#4",.5),("B4",1)],
    ]

    chorus: List[Pattern] = [
        [("D5",.5),("F#5",.5),("A5",1),("G5",.5),("F#5",.5),("E5",1)],
        [("C#5",.5),("D5",.5),("F#5",1),("E5",.5),("D5",.5),("B4",1)],
        [("B4",.5),("C#5",.5),("D5",.5),("F#5",.5),("A5",1),("G5",.5),("E5",.5)],
        [("F#5",.5),("E5",.5),("D5",.5),("C#5",.5),("A#4",.5),("C#5",.5),("B4",1)],
        [("E5",.5),("G5",.5),("B5",1),("A5",.5),("G5",.5),("F#5",1)],
        [("D5",.5),("E5",.5),("G5",1),("F#5",.5),("E5",.5),("C#5",1)],
        [("A4",.5),("B4",.5),("D5",.5),("F#5",.5),("E5",1),("C#5",1)],
        [("D5",.5),("C#5",.5),("B4",1),("A#4",.5),("C#5",.5),("B4",1)],
    ]

    bridge: List[Pattern] = [
        [("B4",1),("D5",.5),("E5",.5),("F#5",1),("A5",1)],
        [("G5",1),("E5",.5),("D5",.5),("C#5",1),("A4",1)],
        [("F#4",.5),("A4",.5),("C#5",1),("B4",.5),("A4",.5),("F#4",1)],
        [("G4",.5),("A4",.5),("B4",1),("C#5",.5),("A#4",.5),("B4",1)],
    ]

    chords = [
        ["B2","F#3","A3","C#4"],
        ["G2","D3","F#3","B3"],
        ["D3","A3","C#4","F#4"],
        ["A2","E3","G3","C#4"],
        ["E3","B3","D4","G4"],
        ["G2","D3","F#3","B3"],
        ["F#2","C#3","E3","B3"],
        ["F#2","C#3","A#3","E4"],
    ]
    roots = ["B1","G1","D2","A1","E2","G1","F#1","F#1"]

    def add_bass_bar(bar: int, dense: bool) -> None:
        start = bar * 4
        idx = bar % 8
        root = roots[idx]
        fifth = nn(root) + 7
        octave = nn(root) + 12
        if dense:
            sequence = [nn(root), fifth, octave, fifth, nn(root), fifth, octave, fifth]
            for step, pitch in enumerate(sequence):
                add_note(events, "Megalo Bass", start + step * .5, pitch, .38, 84 if step % 4 else 100)
        else:
            add_note(events, "Megalo Bass", start, root, 1.7, 96)
            add_note(events, "Megalo Bass", start + 2, root, 1.7, 82)

    def add_guitar_bar(bar: int, strong: bool) -> None:
        start = bar * 4
        idx = bar % 8
        root = nn(roots[idx]) + 12
        chord_notes = [root, root + 7, root + 12]
        if strong:
            for beat in (0, 1.5, 2, 3.5):
                add_chord(events, "Megalo Bg Guitar", start + beat, chord_notes, .34, 74)
            add_chord(events, "Megalo Over Guitar", start, chord_notes, .48, 82)
            add_chord(events, "Megalo Over Guitar", start + 2, chord_notes, .48, 78)
        else:
            add_chord(events, "Hero Nylon Guitar", start, chord_notes, .55, 58)
            add_chord(events, "Hero Nylon Guitar", start + 2, chord_notes, .55, 54)

    def add_drums(bar: int, intensity: int, fill: bool = False) -> None:
        start = bar * 4
        kick_vel = 92 + intensity * 5
        snare_vel = 102 + intensity * 4
        for pos in (0, 2):
            add_note(events, "Hero POWER", start + pos, 36, .09, min(127, kick_vel))
        if intensity >= 1:
            for pos in (1.5, 3.5):
                add_note(events, "Hero POWER", start + pos, 36, .07, 74 + intensity * 4)
        for pos in (1, 3):
            add_note(events, "Hero POWER", start + pos, 38, .10, min(127, snare_vel))
        for step in range(8):
            add_note(events, "Hero POWER", start + step * .5, 42, .055, 46 if step % 2 else 58)
        if intensity >= 2:
            add_note(events, "Hero POWER", start + 2.75, 40, .06, 68)
        if fill:
            for offset, drum_note in [(3.0,45),(3.25,47),(3.5,48),(3.75,50)]:
                add_note(events, "Hero POWER", start + offset, drum_note, .07, 82)

    # 0-3: recognizable motif fragment, not a slow ambient intro.
    for bar in range(4):
        start = bar * 4
        idx = bar % 8
        add_pattern(events, "Megalo Organ 3", start, hook[idx], 78, shift=-12)
        add_pattern(events, "Hero Romantic Tp", start, hook[idx], 96)
        add_chord(events, "Hopes Strings", start, chords[idx], 3.75, 42)
        add_bass_bar(bar, False)
        add_drums(bar, 0, fill=(bar == 3))
        if bar in (0, 2):
            add_chord(events, "Megalo Impact", start, [chords[idx][0],chords[idx][2]], .22, 92)

    # 4-11: complete hook.
    for bar in range(4, 12):
        idx = bar - 4
        start = bar * 4
        add_pattern(events, "Hero Romantic Tp", start, hook[idx], 110)
        add_pattern(events, "Megalo Square 25", start, hook[idx], 46, shift=-12)
        add_chord(events, "Hopes Piano 1", start, chords[idx], .42, 54)
        add_chord(events, "Hopes Piano 1", start + 2, chords[idx], .42, 50)
        add_chord(events, "Hopes Strings", start, chords[idx], 3.76, 48)
        add_bass_bar(bar, True)
        add_guitar_bar(bar, strong=(bar >= 8))
        add_drums(bar, 1, fill=(idx % 4 == 3))
        if idx in (0, 4):
            add_chord(events, "Megalo Impact", start, [chords[idx][0],chords[idx][2],chords[idx][3]], .22, 102)

    # 12-19: melodic chorus, Hopes and Dreams strings/violin with Hero trumpet.
    for bar in range(12, 20):
        idx = bar - 12
        start = bar * 4
        add_pattern(events, "Hopes Violin Detache", start, chorus[idx], 108)
        add_pattern(events, "Hero Romantic Tp", start, chorus[idx], 72, shift=-12)
        add_chord(events, "Hopes Strings", start, chords[idx], 3.78, 60)
        add_chord(events, "Hopes Choir", start, chords[idx], 3.78, 36)
        add_pattern(events, "Hopes Flute", start, [(chords[idx][2],2),(chords[idx][1],2)], 48, shift=12)
        add_bass_bar(bar, True)
        add_guitar_bar(bar, True)
        add_drums(bar, 2, fill=(idx % 4 == 3))
        if idx in (3, 7):
            add_note(events, "Hero Tubular Bells", start + 3.5, nn("B4") + (12 if idx == 7 else 0), .35, 76)

    # 20-23: written bridge, drums half-time and piano foreground.
    for bar in range(20, 24):
        idx = bar - 20
        start = bar * 4
        add_pattern(events, "Hero Piano", start, bridge[idx], 102)
        add_pattern(events, "Hopes Viola", start, bridge[idx], 58, shift=-12)
        add_chord(events, "Hero Strings", start, chords[idx + 4], 3.78, 58)
        add_chord(events, "Hero Choir", start, chords[idx + 4], 3.78, 38)
        add_bass_bar(bar, False)
        add_guitar_bar(bar, False)
        add_drums(bar, 0, fill=(bar == 23))
        add_note(events, "Hero Tubular Bells", start, nn(roots[idx + 4]) + 24, .28, 60)

    # 24-31: hook returns with MEGALOVANIA guitar/brass/organ power.
    for bar in range(24, 32):
        idx = bar - 24
        start = bar * 4
        add_pattern(events, "Megalo Over Guitar", start, hook[idx], 112)
        add_pattern(events, "Hero Romantic Tp", start, hook[idx], 78)
        add_pattern(events, "Megalo Brass", start, hook[idx], 56, shift=-12)
        add_pattern(events, "Megalo Organ 3", start, [(chords[idx][0],2),(chords[idx][2],2)], 54)
        add_chord(events, "Megalo Strings", start, chords[idx], 3.78, 56)
        add_chord(events, "Megalo Ooh Choir", start, chords[idx], 3.78, 32)
        add_bass_bar(bar, True)
        add_guitar_bar(bar, True)
        add_drums(bar, 2, fill=(idx % 4 == 3))
        if idx in (0, 4):
            add_chord(events, "Megalo Impact", start, [chords[idx][0],chords[idx][2],chords[idx][3]], .22, 112)

    # 32-39: final chorus. The melody stays mid-register; high instruments only accent cadences.
    for bar in range(32, 40):
        idx = bar - 32
        start = bar * 4
        add_pattern(events, "Hero Romantic Tp", start, chorus[idx], 116)
        add_pattern(events, "Hopes Violin Detache", start, chorus[idx], 90)
        add_pattern(events, "Megalo Over Guitar", start, chorus[idx], 68, shift=-12)
        add_chord(events, "Hopes Strings", start, chords[idx], 3.78, 66)
        add_chord(events, "Hopes Choir", start, chords[idx], 3.78, 44)
        add_pattern(events, "Hopes Pulse 25", start, [(chords[idx][0],2),(chords[idx][2],2)], 42, shift=12)
        add_bass_bar(bar, True)
        add_guitar_bar(bar, True)
        add_drums(bar, 3, fill=(idx % 2 == 1))
        if idx in (0, 4):
            add_chord(events, "Megalo Impact", start, [chords[idx][0],chords[idx][2],chords[idx][3]], .24, 120)
        if idx in (3, 7):
            add_note(events, "Hero Tubular Bells", start + 3.5, "B5", .34, 72)
            add_note(events, "Hero Piccolo", start + 3.5, "F#6", .22, 42)

    # Strong final B-minor hit.
    end = BARS * 4 - .5
    final_chord = ["B2","F#3","B3","D4","F#4"]
    add_chord(events, "Megalo Impact", end, final_chord, .38, 127)
    add_chord(events, "Megalo Organ 3", end, final_chord, .52, 112)
    add_chord(events, "Hopes Piano 1", end, final_chord, .52, 118)
    add_note(events, "Hero POWER", end, 49, .30, 127)

    return events


def render_stem(soundfont: Path, bank: int, preset: int,
                events: Sequence[Event], gain: float = .78) -> np.ndarray:
    total = int((BARS * BAR + TAIL) * SR)
    synth = fluidsynth.Synth(gain=gain, samplerate=SR)
    sfid = synth.sfload(str(soundfont))
    result = synth.program_select(0, sfid, bank, preset)
    if result != 0:
        synth.delete()
        raise RuntimeError(f"Failed to select exact bank={bank} preset={preset}")
    synth.set_reverb(roomsize=.16, damping=.65, width=.72, level=.09)
    synth.set_chorus(nr=2, level=.14, speed=.26, depth=2.0, type=0)

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
    if cursor < total:
        chunks.append(synth.get_samples(total - cursor))
    synth.delete()

    if not chunks:
        return np.zeros((total, 2), dtype=np.float64)
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


def encode(wav_path: Path, mp3_path: Path, ogg_path: Path | None = None) -> None:
    filt = (
        "highpass=f=28,lowpass=f=12500,"
        "acompressor=threshold=-17dB:ratio=2.0:attack=9:release=115,"
        "alimiter=limit=0.96"
    )
    subprocess.run([
        "ffmpeg","-y","-loglevel","error","-i",str(wav_path),
        "-af",filt,"-codec:a","libmp3lame","-q:a","2",str(mp3_path)
    ], check=True)
    if ogg_path is not None:
        subprocess.run([
            "ffmpeg","-y","-loglevel","error","-i",str(wav_path),
            "-af",filt,"-codec:a","libvorbis","-q:a","6",str(ogg_path)
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
        track = MidiTrack()
        track.append(MetaMessage("track_name", name=name, time=0))
        bank, preset, _ = PRESETS[name]
        while channel == 9:
            channel += 1
        midi_channel = channel % 16
        channel += 1
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
    total = int((BARS * BAR + TAIL) * SR)
    full = np.zeros((total, 2), dtype=np.float64)
    proof = np.zeros_like(full)

    gains = {
        "Hopes Violin Detache": .74,
        "Hopes Piano 1": .52,
        "Hopes POWER": .0,
        "Hopes Glockenspiel": .0,
        "Hopes Flute": .34,
        "Hopes Strings": .50,
        "Hopes Choir": .31,
        "Hopes Viola": .38,
        "Hopes Pulse 25": .30,
        "Hopes Pulse 50": .0,
        "Hopes Duty": .0,
        "Hopes Lead Guitar": .0,
        "Hero Square 25": .0,
        "Hero Duty": .0,
        "Hero Piano": .62,
        "Hero Romantic Tp": .84,
        "Hero POWER": .72,
        "Hero Piano 1": .0,
        "Hero Choir": .34,
        "Hero Piccolo": .18,
        "Hero Violin": .0,
        "Hero Tubular Bells": .34,
        "Hero Strings": .48,
        "Hero Nylon Guitar": .34,
        "Megalo Over Guitar": .64,
        "Megalo Square": .0,
        "Megalo Strings": .48,
        "Megalo Brass": .42,
        "Megalo Saw": .0,
        "Megalo Violin": .0,
        "Megalo Square 25": .30,
        "Megalo Impact": .56,
        "Megalo Ooh Choir": .30,
        "Megalo Organ 3": .47,
        "Megalo Rock Organ": .0,
        "Megalo Bg Guitar": .44,
        "Megalo Bass": .66,
    }

    proof_tracks = {
        "Hero Romantic Tp": .92,
        "Hopes Violin Detache": .76,
        "Hopes Piano 1": .52,
        "Hopes Strings": .42,
        "Megalo Bass": .42,
    }

    selected_lines = [
        f"BPM: {BPM}",
        "Only exact track-group presets used; no generic GM fallback.",
        "",
    ]

    for name, track_events in events.items():
        if not track_events:
            continue
        bank, preset, source = PRESETS[name]
        selected_lines.append(f"{name}: bank={bank} preset={preset} | {source}")
        stem = render_stem(args.soundfont, bank, preset, track_events)
        length = min(total, len(stem))
        gain = gains.get(name, 0.0)
        if gain:
            full[:length] += stem[:length] * gain
        if name in proof_tracks:
            # Proof is limited to the first 20 bars: intro + hook + chorus start.
            proof_length = min(length, int((20 * BAR + .8) * SR))
            proof[:proof_length] += stem[:proof_length] * proof_tracks[name]

    full = np.tanh(full * 1.06)
    proof = np.tanh(proof * 1.02)

    # Trim proof to 20 bars instead of padding it to full track length.
    proof = proof[:int((20 * BAR + 1.2) * SR)]

    full_wav = args.out / "KRIS_RED_VERDICT_FULL.wav"
    proof_wav = args.out / "KRIS_RED_VERDICT_MELODY_PROOF.wav"
    write_wav(full_wav, full)
    write_wav(proof_wav, proof)

    encode(full_wav,
           args.out / "KRIS_RED_VERDICT_FULL.mp3",
           args.out / "KRIS_RED_VERDICT_GAME.ogg")
    encode(proof_wav, args.out / "KRIS_RED_VERDICT_MELODY_PROOF.mp3")
    export_midi(args.out / "KRIS_RED_VERDICT.mid", events)
    (args.out / "KRIS_RED_VERDICT_INSTRUMENTS.txt").write_text(
        "\n".join(selected_lines) + "\n", encoding="utf-8"
    )

    print("\n".join(selected_lines))


if __name__ == "__main__":
    main()
