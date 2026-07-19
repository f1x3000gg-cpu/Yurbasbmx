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
BPM = 160
BEAT = 60.0 / BPM
BAR_SECONDS = 4.0 * BEAT
BARS = 48
TAIL_SECONDS = 2.4
TPB = 480

PITCH = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

Event = Tuple[float, float, int, int]
Pattern = Sequence[Tuple[str | int | None, float]]

# Exact preset locations in UNDERTALE Soundfont 2. Every active preset below
# belongs to a named, popular UNDERTALE track group. No saxophone and no
# generic GM fallback are used.
PRESETS = {
    # 046 Spear of Justice
    "Spear Romantic Tp": (1, 55, "046 Spear of Justice - Romantic Tp"),
    "Spear Brass Tp":    (1, 57, "046 Spear of Justice - Brass Trumpet"),
    "Spear Orch Hit":    (1, 58, "046 Spear of Justice - Orchestra Hit"),
    "Spear Choir":       (1, 59, "046 Spear of Justice - Ahh Choir"),
    "Spear Strings":     (1, 60, "046 Spear of Justice - Strings"),
    "Spear Piano":       (1, 61, "046 Spear of Justice - Piano"),
    "Spear Drums":       (1, 62, "046 Spear of Justice - Drumkit"),
    "Spear Pulse25":     (1, 65, "046 Spear of Justice - Pulse 25%"),
    "Spear Triangle":    (1, 66, "046 Spear of Justice - Triangle"),

    # 059 Spider Dance
    "Spider Strings":     (1, 126, "059 Spider Dance - Strings"),
    "Spider Harpsichord": (2,   0, "059 Spider Dance - Harpsichord"),
    "Spider ChoirHorn":   (2,   1, "059 Spider Dance - Choir+Horn"),

    # 065 CORE
    "Core Choir":  (2, 32, "065 CORE - Ahh Choir"),
    "Core Strings":(2, 33, "065 CORE - Strings"),
    "Core Glock":  (2, 34, "065 CORE - Glockenspiel"),
    "Core Brass":  (2, 35, "065 CORE - Brass"),
    "Core Drums":  (2, 36, "065 CORE - Drumkit"),
    "Core Piano":  (2, 37, "065 CORE - Piano"),
    "Core Saw":    (2, 38, "065 CORE - Saw"),

    # 068 Death by Glamour, saxophone preset intentionally excluded
    "Glamour Piano":  (2, 42, "068 Death by Glamour - Piano"),
    "Glamour Strings":(2, 43, "068 Death by Glamour - Strings"),
    "Glamour Drums":  (2, 45, "068 Death by Glamour - Drumkit"),
    "Glamour Hit":    (2, 46, "068 Death by Glamour - Impact Hit"),
    "Glamour Rhodes": (2, 47, "068 Death by Glamour - Rhodes EP"),

    # 077 ASGORE
    "Asgore Piano":       (2, 64, "077 ASGORE - Piano"),
    "Asgore Glock":       (2, 70, "077 ASGORE - Glockenspiel"),
    "Asgore Tubular":     (2, 71, "077 ASGORE - Tubular Bells"),
    "Asgore Violin":      (2, 73, "077 ASGORE - Violin"),
    "Asgore Timpani":     (2, 74, "077 ASGORE - Timpani"),
    "Asgore Strings":     (2, 75, "077 ASGORE - Strings"),
    "Asgore Choir":       (2, 76, "077 ASGORE - Choir Aahs"),
    "Asgore Hit":         (2, 77, "077 ASGORE - Orchestra Hit"),
    "Asgore Brass":       (2, 78, "077 ASGORE - Brass"),
    "Asgore Drums":       (2, 80, "077 ASGORE - Drumkit"),

    # 080 Finale
    "Finale Piano1":      (2, 93, "080 Finale - Piano 1"),
    "Finale Choir":       (2, 94, "080 Finale - Choir Aahs"),
    "Finale Glock":       (2, 95, "080 Finale - Glockenspiel"),
    "Finale Tubular":     (2, 96, "080 Finale - Tubular Bells"),
    "Finale Strings":     (2, 97, "080 Finale - Strings"),
    "Finale Trumpet":     (2, 98, "080 Finale - Trumpet"),
    "Finale StringsMarc": (2, 99, "080 Finale - Strings marc"),
    "Finale SuperSaw":    (2,100, "080 Finale - SuperSawA"),

    # 087 Hopes and Dreams
    "Hopes Violin":   (2,125, "087 Hopes and Dreams - Violin Detache"),
    "Hopes Piano1":   (2,126, "087 Hopes and Dreams - Piano 1"),
    "Hopes POWER":    (2,127, "087 Hopes and Dreams - POWER DrumKit"),
    "Hopes Strings":  (3,  2, "087 Hopes and Dreams - Strings"),
    "Hopes Choir":    (3,  3, "087 Hopes and Dreams - Choir Aahs"),

    # 098 Battle Against a True Hero
    "Hero Romantic":  (3, 65, "098 Battle Against a True Hero - Romantic Tp"),
    "Hero POWER":     (3, 66, "098 Battle Against a True Hero - POWER DrumKit"),
    "Hero Piano1":    (3, 67, "098 Battle Against a True Hero - Piano 1"),
    "Hero Choir":     (3, 68, "098 Battle Against a True Hero - Choir Aahs"),
    "Hero Violin":    (3, 70, "098 Battle Against a True Hero - Violin"),
    "Hero Tubular":   (3, 71, "098 Battle Against a True Hero - Tubular Bells"),
    "Hero Strings":   (3, 72, "098 Battle Against a True Hero - Strings"),

    # 099 Power of NEO
    "Neo Strings": (3, 74, "099 Power of NEO - Strings"),
    "Neo Hit":     (3, 75, "099 Power of NEO - Impact Hit"),
    "Neo Organ":   (3, 76, "099 Power of NEO - Organ 3"),
    "Neo OverGt":  (3, 77, "099 Power of NEO - Overdrive Guitar"),
    "Neo DistGt":  (3, 78, "099 Power of NEO - Distortion Guitar"),
}


def note_number(note: str | int) -> int:
    if isinstance(note, int):
        return note
    if len(note) >= 3 and note[1] in "#b":
        pitch, octave = note[:2], int(note[2:])
    else:
        pitch, octave = note[0], int(note[1:])
    return 12 * (octave + 1) + PITCH[pitch]


def add_note(events: Dict[str, List[Event]], track: str, start: float,
             note: str | int, duration: float, velocity: int) -> None:
    events[track].append((float(start), float(duration), note_number(note), int(velocity)))


def add_pattern(events: Dict[str, List[Event]], track: str, start: float,
                pattern: Pattern, velocity: int, shift: int = 0) -> None:
    cursor = float(start)
    for note, duration in pattern:
        if note is not None:
            add_note(events, track, cursor, note_number(note) + shift,
                     duration * .94, velocity)
        cursor += duration


def add_chord(events: Dict[str, List[Event]], track: str, start: float,
              notes: Sequence[str | int], duration: float, velocity: int) -> None:
    for note in notes:
        add_note(events, track, start, note, duration, velocity)


def build_events() -> Dict[str, List[Event]]:
    events: Dict[str, List[Event]] = {name: [] for name in PRESETS}

    # D minor. The lead was written as four coherent phrases before arranging.
    # It uses repetition, answer phrases and cadence notes; there is no random
    # note generator and no melodic sixteenth-note stream.
    theme_a: List[Pattern] = [
        [(None,.5),("A4",.5),("D5",1),("F5",.5),("E5",.5),("D5",1)],
        [("C5",.5),("A4",.5),("G4",1),("A4",.5),("C5",.5),("D5",1)],
        [(None,.5),("F5",.5),("A5",1),("G5",.5),("F5",.5),("E5",1)],
        [("D5",.5),("C5",.5),("A4",1),("C#5",.5),("E5",.5),("D5",1)],
        [("A4",.5),("D5",.5),("F5",1),("A5",.5),("G5",.5),("F5",1)],
        [("E5",.5),("D5",.5),("C5",1),("A4",.5),("C5",.5),("D5",1)],
        [("F5",1),("E5",.5),("D5",.5),("C5",1),("A4",1)],
        [("Bb4",.5),("C5",.5),("D5",1),("E5",.5),("C#5",.5),("D5",1)],
    ]

    answer_b: List[Pattern] = [
        [("D5",1),("F5",.5),("E5",.5),("D5",1),("A4",1)],
        [("Bb4",.5),("C5",.5),("D5",1),("F5",1),("E5",1)],
        [("G5",.5),("F5",.5),("E5",1),("D5",.5),("C5",.5),("A4",1)],
        [("C#5",.5),("D5",.5),("E5",1),("G5",.5),("F5",.5),("E5",1)],
        [("F5",.5),("A5",.5),("Bb5",1),("A5",.5),("G5",.5),("F5",1)],
        [("E5",.5),("D5",.5),("C5",1),("A4",.5),("C5",.5),("D5",1)],
        [("G5",1),("F5",.5),("E5",.5),("D5",1),("A4",1)],
        [("Bb4",.5),("C5",.5),("D5",1),("E5",.5),("C#5",.5),("D5",1)],
    ]

    chorus: List[Pattern] = [
        [("D5",.5),("F5",.5),("A5",1),("A5",.5),("G5",.5),("F5",1)],
        [("G5",.5),("F5",.5),("E5",1),("D5",2)],
        [("F5",.5),("A5",.5),("Bb5",1),("A5",.5),("G5",.5),("F5",1)],
        [("E5",.5),("D5",.5),("C#5",1),("D5",2)],
        [("A4",.5),("D5",.5),("F5",1),("E5",.5),("F5",.5),("G5",1)],
        [("A5",.5),("G5",.5),("F5",1),("E5",.5),("D5",.5),("C5",1)],
        [("Bb4",.5),("D5",.5),("F5",1),("E5",.5),("C#5",.5),("D5",1)],
        [("D5",2),("A4",.5),("C#5",.5),("D5",1)],
    ]

    bridge: List[Pattern] = [
        [("D5",2),("F5",1),("E5",1)],
        [("C5",2),("A4",1),("D5",1)],
        [("Bb4",2),("D5",1),("F5",1)],
        [("E5",1),("C#5",1),("D5",2)],
    ]

    chords = [
        ["D3","A3","C4","F4"],
        ["Bb2","F3","A3","D4"],
        ["F3","C4","E4","A4"],
        ["C3","G3","Bb3","E4"],
        ["G2","D3","F3","Bb3"],
        ["Bb2","F3","A3","D4"],
        ["A2","E3","G3","C#4"],
        ["A2","E3","G3","C#4"],
    ]
    chorus_chords = [
        ["D3","A3","C4","F4"],
        ["Bb2","F3","A3","D4"],
        ["F3","C4","E4","A4"],
        ["C3","G3","Bb3","E4"],
        ["Bb2","F3","A3","D4"],
        ["C3","G3","Bb3","E4"],
        ["D3","A3","C4","F4"],
        ["A2","E3","G3","C#4"],
    ]
    roots = ["D2","Bb1","F2","C2","G1","Bb1","A1","A1"]
    chorus_roots = ["D2","Bb1","F2","C2","Bb1","C2","D2","A1"]

    def bass_bar(bar: int, dense: bool, chorus_mode: bool = False) -> None:
        start = bar * 4
        roots_used = chorus_roots if chorus_mode else roots
        root = note_number(roots_used[bar % 8])
        fifth = root + 7
        octave = root + 12
        track = "Spear Triangle"
        if dense:
            pitches = [root, fifth, octave, fifth, root, fifth, octave, fifth]
            for step, pitch in enumerate(pitches):
                add_note(events, track, start + step * .5, pitch, .39,
                         88 if step % 4 else 102)
        else:
            add_note(events, track, start, root, 1.75, 98)
            add_note(events, track, start + 2, fifth, 1.75, 78)

    def rhythm_guitar(bar: int, strong: bool, chorus_mode: bool = False) -> None:
        start = bar * 4
        chord_set = chorus_chords if chorus_mode else chords
        notes = [note_number(n) - 12 for n in chord_set[bar % 8][:3]]
        positions = (0, 1.5, 2, 3.5) if strong else (0, 2)
        for pos in positions:
            add_chord(events, "Neo OverGt", start + pos, notes,
                      .34 if strong else .48, 74 if strong else 58)
        if strong:
            for pos in (0, 2):
                add_chord(events, "Neo DistGt", start + pos, notes, .42, 66)

    def drum_bar(bar: int, kit: str, intensity: int, fill: bool = False) -> None:
        start = bar * 4
        kick = 36
        snare = 38
        closed_hat = 42
        crash = 49
        for pos in (0, 2):
            add_note(events, kit, start + pos, kick, .08, 94 + intensity * 5)
        if intensity >= 1:
            for pos in (1.5, 3.5):
                add_note(events, kit, start + pos, kick, .07, 70 + intensity * 5)
        for pos in (1, 3):
            add_note(events, kit, start + pos, snare, .10, 104 + intensity * 4)
        for step in range(8):
            add_note(events, kit, start + step * .5, closed_hat, .055,
                     46 if step % 2 else 58)
        if bar in (0, 4, 12, 20, 32, 40):
            add_note(events, kit, start, crash, .20, 102)
        if fill:
            for offset, drum_note in ((3.0,45),(3.25,47),(3.5,48),(3.75,50)):
                add_note(events, kit, start + offset, drum_note, .07, 78 + intensity * 4)

    def chord_pad(track: str, bar: int, velocity: int, chorus_mode: bool = False) -> None:
        chord_set = chorus_chords if chorus_mode else chords
        add_chord(events, track, bar * 4, chord_set[bar % 8], 3.78, velocity)

    # 0-3: immediate melodic identity, no slow ambient intro.
    for bar in range(4):
        start = bar * 4
        add_pattern(events, "Asgore Piano", start, theme_a[bar], 94)
        add_pattern(events, "Spear Romantic Tp", start, theme_a[bar], 72)
        chord_pad("Asgore Strings", bar, 42)
        bass_bar(bar, False)
        drum_bar(bar, "Spear Drums", 0, fill=(bar == 3))
        if bar in (0, 2):
            add_chord(events, "Asgore Hit", start,
                      [chords[bar][0], chords[bar][2]], .22, 84)

    # 4-11: full verse/theme A.
    for bar in range(4, 12):
        idx = bar - 4
        start = bar * 4
        add_pattern(events, "Spear Romantic Tp", start, theme_a[idx], 108)
        add_pattern(events, "Hopes Piano1", start, theme_a[idx], 52, shift=-12)
        chord_pad("Spear Strings", bar, 50)
        add_chord(events, "Spear Piano", start, chords[idx], .34, 52)
        add_chord(events, "Spear Piano", start + 2, chords[idx], .34, 48)
        bass_bar(bar, bar >= 8)
        drum_bar(bar, "Spear Drums", 1, fill=(idx % 4 == 3))
        if idx in (0, 4):
            add_chord(events, "Spear Orch Hit", start,
                      [chords[idx][0],chords[idx][2],chords[idx][3]], .22, 100)

    # 12-19: answer phrase with Spider Dance/CORE colours, but the melody stays clear.
    for bar in range(12, 20):
        idx = bar - 12
        start = bar * 4
        add_pattern(events, "Hopes Violin", start, answer_b[idx], 106)
        add_pattern(events, "Finale Piano1", start, answer_b[idx], 48, shift=-12)
        chord_pad("Spider Strings", bar, 48)
        if idx % 2 == 0:
            add_chord(events, "Spider ChoirHorn", start, chords[idx], 1.75, 30)
        # Harpsichord is rhythmic support, not a second random melody.
        for beat, chord_note in zip((0,.5,1,1.5,2,2.5,3,3.5),
                                    chords[idx] * 2):
            add_note(events, "Spider Harpsichord", start + beat,
                     note_number(chord_note) + 12, .30, 42)
        bass_bar(bar, True)
        drum_bar(bar, "Core Drums", 1, fill=(idx % 4 == 3))
        if idx in (3, 7):
            add_note(events, "Core Glock", start + 3.5, "D6", .22, 42)

    # 20-27: first chorus. Broad, singable lead and stronger harmony.
    for bar in range(20, 28):
        idx = bar - 20
        start = bar * 4
        add_pattern(events, "Hero Romantic", start, chorus[idx], 112)
        add_pattern(events, "Hopes Violin", start, chorus[idx], 82)
        add_pattern(events, "Finale StringsMarc", start,
                    [(chorus_chords[idx][0],1),(chorus_chords[idx][2],1),
                     (chorus_chords[idx][1],1),(chorus_chords[idx][2],1)], 52)
        chord_pad("Hopes Strings", bar, 60, chorus_mode=True)
        chord_pad("Hopes Choir", bar, 34, chorus_mode=True)
        add_pattern(events, "Neo Organ", start,
                    [(chorus_chords[idx][0],2),(chorus_chords[idx][2],2)], 44)
        bass_bar(bar, True, chorus_mode=True)
        rhythm_guitar(bar, True, chorus_mode=True)
        drum_bar(bar, "Hero POWER", 2, fill=(idx % 4 == 3))
        if idx in (0, 4):
            add_chord(events, "Neo Hit", start,
                      [chorus_chords[idx][0],chorus_chords[idx][2],
                       chorus_chords[idx][3]], .24, 110)

    # 28-31: short emotional bridge; no drum or note spam.
    for bar in range(28, 32):
        idx = bar - 28
        start = bar * 4
        add_pattern(events, "Glamour Rhodes", start, bridge[idx], 104)
        add_pattern(events, "Asgore Violin", start, bridge[idx], 62, shift=-12)
        chord_pad("Asgore Strings", bar, 58)
        chord_pad("Asgore Choir", bar, 30)
        bass_bar(bar, False)
        drum_bar(bar, "Glamour Drums", 0, fill=(bar == 31))
        add_note(events, "Asgore Timpani", start,
                 note_number(roots[idx + 4]) + 12, .48, 72)

    # 32-39: chorus returns with Finale colour and controlled synth power.
    for bar in range(32, 40):
        idx = bar - 32
        start = bar * 4
        add_pattern(events, "Finale Trumpet", start, chorus[idx], 112)
        add_pattern(events, "Hopes Violin", start, chorus[idx], 92)
        add_pattern(events, "Hero Romantic", start, chorus[idx], 58, shift=-12)
        chord_pad("Finale Strings", bar, 62, chorus_mode=True)
        chord_pad("Finale Choir", bar, 38, chorus_mode=True)
        add_pattern(events, "Finale SuperSaw", start,
                    [(chorus_chords[idx][0],2),(chorus_chords[idx][2],2)], 34)
        bass_bar(bar, True, chorus_mode=True)
        rhythm_guitar(bar, True, chorus_mode=True)
        drum_bar(bar, "Hopes POWER", 2, fill=(idx % 4 == 3))
        if idx in (0, 4):
            add_chord(events, "Glamour Hit", start,
                      [chorus_chords[idx][0],chorus_chords[idx][2],
                       chorus_chords[idx][3]], .23, 106)
        if idx in (3, 7):
            add_note(events, "Finale Tubular", start + 3.5, "D5", .32, 60)

    # 40-47: final theme A as a victorious reprise, then a clean D-minor ending.
    for bar in range(40, 48):
        idx = bar - 40
        start = bar * 4
        add_pattern(events, "Spear Brass Tp", start, theme_a[idx], 112)
        add_pattern(events, "Hopes Violin", start, theme_a[idx], 90)
        add_pattern(events, "Core Brass", start, theme_a[idx], 46, shift=-12)
        chord_pad("Hero Strings", bar, 64)
        chord_pad("Hero Choir", bar, 38)
        add_pattern(events, "Core Saw", start,
                    [(chords[idx][0],2),(chords[idx][2],2)], 30)
        bass_bar(bar, True)
        rhythm_guitar(bar, True)
        drum_bar(bar, "Hero POWER", 3, fill=(idx % 4 == 3))
        if idx in (0, 4):
            add_chord(events, "Asgore Hit", start,
                      [chords[idx][0],chords[idx][2],chords[idx][3]], .24, 114)
        if idx in (3, 7):
            add_note(events, "Asgore Tubular", start + 3.5, "D5", .30, 64)

    end = BARS * 4 - .5
    final_chord = ["D2","A2","D3","F3","A3","D4"]
    add_chord(events, "Asgore Hit", end, final_chord, .42, 127)
    add_chord(events, "Neo Organ", end, final_chord, .58, 108)
    add_chord(events, "Hopes Piano1", end, final_chord, .58, 116)
    add_chord(events, "Hopes Strings", end, final_chord, .70, 96)
    add_note(events, "Hero POWER", end, 49, .30, 127)

    return events


def render_stem(soundfont: Path, bank: int, preset: int,
                events: Sequence[Event], gain: float = .78) -> np.ndarray:
    total_frames = int((BARS * BAR_SECONDS + TAIL_SECONDS) * SR)
    synth = fluidsynth.Synth(gain=gain, samplerate=SR)
    sfid = synth.sfload(str(soundfont))
    result = synth.program_select(0, sfid, bank, preset)
    if result != 0:
        synth.delete()
        raise RuntimeError(f"Failed exact preset bank={bank} preset={preset}")
    synth.set_reverb(roomsize=.16, damping=.66, width=.72, level=.09)
    synth.set_chorus(nr=2, level=.13, speed=.25, depth=2.0, type=0)

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
    audio_filter = (
        "highpass=f=28,lowpass=f=12500,"
        "acompressor=threshold=-17dB:ratio=2.0:attack=10:release=120,"
        "alimiter=limit=0.96"
    )
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path),
        "-af", audio_filter, "-codec:a", "libmp3lame", "-q:a", "2", str(mp3_path)
    ], check=True)
    if ogg_path is not None:
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path),
            "-af", audio_filter, "-codec:a", "libvorbis", "-q:a", "6", str(ogg_path)
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

    events = build_events()
    total_frames = int((BARS * BAR_SECONDS + TAIL_SECONDS) * SR)
    full = np.zeros((total_frames, 2), dtype=np.float64)
    proof = np.zeros_like(full)

    gains = {
        "Spear Romantic Tp": .78, "Spear Brass Tp": .72, "Spear Orch Hit": .50,
        "Spear Choir": .28, "Spear Strings": .47, "Spear Piano": .46,
        "Spear Drums": .62, "Spear Pulse25": .24, "Spear Triangle": .62,
        "Spider Strings": .43, "Spider Harpsichord": .31, "Spider ChoirHorn": .26,
        "Core Choir": .28, "Core Strings": .42, "Core Glock": .18,
        "Core Brass": .34, "Core Drums": .62, "Core Piano": .48, "Core Saw": .22,
        "Glamour Piano": .44, "Glamour Strings": .42, "Glamour Drums": .58,
        "Glamour Hit": .45, "Glamour Rhodes": .68,
        "Asgore Piano": .66, "Asgore Glock": .18, "Asgore Tubular": .28,
        "Asgore Violin": .40, "Asgore Timpani": .42, "Asgore Strings": .50,
        "Asgore Choir": .30, "Asgore Hit": .52, "Asgore Brass": .34,
        "Asgore Drums": .58,
        "Finale Piano1": .48, "Finale Choir": .32, "Finale Glock": .16,
        "Finale Tubular": .28, "Finale Strings": .50, "Finale Trumpet": .78,
        "Finale StringsMarc": .42, "Finale SuperSaw": .24,
        "Hopes Violin": .72, "Hopes Piano1": .52, "Hopes POWER": .68,
        "Hopes Strings": .52, "Hopes Choir": .34,
        "Hero Romantic": .82, "Hero POWER": .72, "Hero Piano1": .48,
        "Hero Choir": .32, "Hero Violin": .44, "Hero Tubular": .28,
        "Hero Strings": .50,
        "Neo Strings": .42, "Neo Hit": .50, "Neo Organ": .40,
        "Neo OverGt": .48, "Neo DistGt": .38,
    }

    proof_tracks = {
        "Asgore Piano": .72,
        "Spear Romantic Tp": .88,
        "Hopes Violin": .82,
        "Hero Romantic": .88,
        "Finale Trumpet": .82,
        "Hopes Piano1": .44,
        "Spear Strings": .34,
        "Hopes Strings": .38,
        "Finale Strings": .38,
        "Spear Triangle": .28,
    }

    selected_lines = [
        "KRIS // LAST PROMISE",
        f"BPM: {BPM}",
        "Original melody: theme A, answer B, chorus, bridge, final reprise.",
        "No saxophone. No generic GM fallback. No random note generator.",
        "",
    ]

    for name, track_events in events.items():
        if not track_events:
            continue
        bank, preset, source = PRESETS[name]
        selected_lines.append(f"{name}: bank={bank} preset={preset} | {source}")
        stem = render_stem(args.soundfont, bank, preset, track_events)
        length = min(total_frames, len(stem))
        gain = gains.get(name, 0.0)
        if gain:
            full[:length] += stem[:length] * gain
        if name in proof_tracks:
            proof_end = int((28 * BAR_SECONDS + 1.0) * SR)
            proof_length = min(length, proof_end)
            proof[:proof_length] += stem[:proof_length] * proof_tracks[name]

    full = np.tanh(full * 1.04)
    proof = np.tanh(proof * 1.01)
    proof = proof[:int((28 * BAR_SECONDS + 1.2) * SR)]

    full_wav = args.out / "KRIS_LAST_PROMISE_FULL.wav"
    proof_wav = args.out / "KRIS_LAST_PROMISE_MELODY.wav"
    write_wav(full_wav, full)
    write_wav(proof_wav, proof)

    encode(full_wav,
           args.out / "KRIS_LAST_PROMISE_FULL.mp3",
           args.out / "KRIS_LAST_PROMISE_GAME.ogg")
    encode(proof_wav, args.out / "KRIS_LAST_PROMISE_MELODY.mp3")
    export_midi(args.out / "KRIS_LAST_PROMISE.mid", events)
    (args.out / "KRIS_LAST_PROMISE_INSTRUMENTS.txt").write_text(
        "\n".join(selected_lines) + "\n", encoding="utf-8"
    )
    print("\n".join(selected_lines))


if __name__ == "__main__":
    main()
