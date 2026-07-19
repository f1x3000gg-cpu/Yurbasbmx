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
BEAT = 60.0 / BPM
BARS = 8
TAIL = 2.2
TPB = 480

PITCH = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

Event = Tuple[float, float, int, int]
Pattern = Sequence[Tuple[str | int | None, float]]

# Exact presets from the UNDERTALE Soundfont 2 track groups.
# Track 015 is sans.; track 087 is Hopes and Dreams.
PRESETS = {
    "Sans Clavinet": (0, 16, "015 sans. - Clavinet"),
    "Sans Bass": (0, 35, "015 sans. - Bass"),
    "Sans Drums": (0, 17, "015 sans. - Drums"),
    "Asriel Piano": (2, 126, "087 Hopes and Dreams - Piano 1"),
    "Asriel POWER": (2, 127, "087 Hopes and Dreams - POWER DrumKit"),
    "Asriel Choir": (3, 3, "087 Hopes and Dreams - Choir Aahs"),
    "Asriel Pulse25": (3, 5, "087 Hopes and Dreams - Pulse 25%"),
    "Asriel Lead Guitar": (3, 8, "087 Hopes and Dreams - Lead Guitar"),
}


def nn(note: str | int) -> int:
    if isinstance(note, int):
        return note
    if len(note) >= 3 and note[1] in "#b":
        name, octave = note[:2], int(note[2:])
    else:
        name, octave = note[0], int(note[1:])
    return 12 * (octave + 1) + PITCH[name]


def add(events: Dict[str, List[Event]], track: str, start: float,
        note: str | int, duration: float, velocity: int) -> None:
    events[track].append((start, duration, nn(note), velocity))


def pattern(events: Dict[str, List[Event]], track: str, start: float,
            notes: Pattern, velocity: int, shift: int = 0) -> None:
    cursor = start
    for note, duration in notes:
        if note is not None:
            add(events, track, cursor, nn(note) + shift, duration * 0.93, velocity)
        cursor += duration


def chord(events: Dict[str, List[Event]], track: str, start: float,
          notes: Sequence[str], duration: float, velocity: int) -> None:
    for note in notes:
        add(events, track, start, note, duration, velocity)


def make_events() -> Dict[str, List[Event]]:
    events = {name: [] for name in PRESETS}

    # One manually written eight-bar G-minor melody. No random generator.
    melody: List[Pattern] = [
        [(None,.5),("D4",.5),("G4",1),("Bb4",.5),("A4",.5),("G4",1)],
        [("F4",.5),("G4",.5),("D5",1),("C5",.5),("Bb4",.5),("A4",1)],
        [("G4",.5),("Bb4",.5),("D5",1),("C5",.5),("Bb4",.5),("G4",1)],
        [("A4",.5),("Bb4",.5),("C5",1),("D5",1),("F#4",1)],
        [("D5",.5),("Eb5",.5),("F5",1),("D5",.5),("C5",.5),("Bb4",1)],
        [("A4",.5),("G4",.5),("Bb4",1),("D5",.5),("C5",.5),("A4",1)],
        [("G4",1),("Bb4",.5),("C5",.5),("D5",1),("F5",1)],
        [("Eb5",.5),("D5",.5),("C5",1),("A4",.5),("F#4",.5),("G4",1)],
    ]

    chords = [
        ["G2","D3","G3","Bb3"],
        ["Eb2","Bb2","G3","Bb3"],
        ["Bb2","F3","Bb3","D4"],
        ["D2","A2","C3","F#3"],
        ["Cm2","G2","C3","Eb3"],
        ["Eb2","Bb2","G3","Bb3"],
        ["F2","C3","F3","A3"],
        ["D2","A2","C3","F#3"],
    ]
    roots = ["G1","Eb1","Bb1","D1","C2","Eb1","F1","D1"]

    for bar in range(BARS):
        start = bar * 4.0
        m = melody[bar]

        # Main melody is the actual Hopes and Dreams lead-guitar preset.
        pattern(events, "Asriel Lead Guitar", start, m, 110)
        pattern(events, "Asriel Pulse25", start, m, 40, shift=-12)

        # Piano and choir provide Asriel's serious harmonic weight.
        chord(events, "Asriel Piano", start, chords[bar], 0.42, 62)
        chord(events, "Asriel Piano", start + 2.0, chords[bar], 0.42, 56)
        chord(events, "Asriel Choir", start, chords[bar], 3.75, 31)

        # sans. rhythm section: exact clavinet, bass and drums, no saxophone.
        root = nn(roots[bar])
        fifth = root + 7
        octave = root + 12
        bass_seq = [root, fifth, octave, fifth, root, fifth, octave, fifth]
        for step, pitch in enumerate(bass_seq):
            add(events, "Sans Bass", start + step * .5, pitch, .38,
                88 if step % 4 else 102)

        clav_notes = [nn(chords[bar][1]), nn(chords[bar][2]), nn(chords[bar][3])]
        for pos, pitch in zip((.5,1.5,2.5,3.5),
                              (clav_notes[0],clav_notes[1],clav_notes[2],clav_notes[1])):
            add(events, "Sans Clavinet", start + pos, pitch, .24, 66)

        # Controlled battle groove: no breakcore spam.
        for pos in (0, 2):
            add(events, "Asriel POWER", start + pos, 36, .08, 108)
        for pos in (1, 3):
            add(events, "Asriel POWER", start + pos, 38, .09, 112)
        for step in range(8):
            add(events, "Asriel POWER", start + step * .5, 42, .045,
                42 if step % 2 else 55)
        for pos in (0.75, 2.75):
            add(events, "Sans Drums", start + pos, 35, .06, 48)

        if bar in (3, 7):
            for offset, note in ((3.0,45),(3.25,47),(3.5,48),(3.75,50)):
                add(events, "Asriel POWER", start + offset, note, .06, 78)

    return events


def render(soundfont: Path, bank: int, preset: int,
           events: Sequence[Event], total_frames: int) -> np.ndarray:
    synth = fluidsynth.Synth(gain=.8, samplerate=SR)
    sfid = synth.sfload(str(soundfont))
    if synth.program_select(0, sfid, bank, preset) != 0:
        synth.delete()
        raise RuntimeError(f"Cannot select bank={bank} preset={preset}")
    synth.set_reverb(roomsize=.18, damping=.7, width=.65, level=.08)
    synth.set_chorus(nr=2, level=.12, speed=.25, depth=1.8, type=0)

    timeline = []
    for start, duration, note, velocity in events:
        timeline.append((int(start * BEAT * SR), 1, note, velocity))
        timeline.append((int((start + duration) * BEAT * SR), 0, note, 0))
    timeline.sort(key=lambda x: (x[0], x[1]))

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
    with wave.open(str(path), "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes(pcm.tobytes())


def encode(wav: Path, mp3: Path) -> None:
    subprocess.run([
        "ffmpeg","-y","-loglevel","error","-i",str(wav),
        "-af","highpass=f=28,lowpass=f=13000,acompressor=threshold=-18dB:ratio=2:attack=10:release=120,alimiter=limit=.96",
        "-codec:a","libmp3lame","-q:a","2",str(mp3)
    ], check=True)


def export_midi(path: Path, events: Dict[str, List[Event]]) -> None:
    midi = MidiFile(ticks_per_beat=TPB)
    tempo = MidiTrack()
    tempo.append(MetaMessage("track_name", name="Tempo", time=0))
    tempo.append(MetaMessage("set_tempo", tempo=bpm2tempo(BPM), time=0))
    midi.tracks.append(tempo)

    channel = 0
    for name, evs in events.items():
        if not evs:
            continue
        while channel == 9:
            channel += 1
        ch = channel % 16
        channel += 1
        bank, preset, _ = PRESETS[name]
        track = MidiTrack()
        track.append(MetaMessage("track_name", name=name, time=0))
        track.append(Message("control_change", channel=ch, control=0,
                             value=min(127, bank), time=0))
        track.append(Message("program_change", channel=ch,
                             program=min(127, preset), time=0))
        timeline = []
        for start, duration, note, velocity in evs:
            timeline.append((round(start * TPB), 1, note, velocity))
            timeline.append((round((start + duration) * TPB), 0, note, 0))
        timeline.sort(key=lambda x: (x[0], x[1]))
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

    events = make_events()
    total = int((BARS * 4 * BEAT + TAIL) * SR)
    context = np.zeros((total, 2), dtype=np.float64)
    melody = np.zeros_like(context)

    context_gains = {
        "Sans Clavinet": .48,
        "Sans Bass": .67,
        "Sans Drums": .28,
        "Asriel Piano": .48,
        "Asriel POWER": .72,
        "Asriel Choir": .27,
        "Asriel Pulse25": .28,
        "Asriel Lead Guitar": .88,
    }
    melody_gains = {
        "Asriel Lead Guitar": .92,
        "Asriel Piano": .40,
        "Asriel Pulse25": .20,
    }

    report = [
        f"BPM: {BPM}",
        "Exact track groups only: 015 sans. and 087 Hopes and Dreams.",
        "No saxophone, trumpet, clarinet, flute, violin, brass or strings.",
        "No generic GM fallback and no random-note generator.",
        "",
    ]

    for name, evs in events.items():
        if not evs:
            continue
        bank, preset, source = PRESETS[name]
        report.append(f"{name}: bank={bank} preset={preset} | {source}")
        stem = render(args.soundfont, bank, preset, evs, total)
        n = min(total, len(stem))
        context[:n] += stem[:n] * context_gains.get(name, 0)
        melody[:n] += stem[:n] * melody_gains.get(name, 0)

    context = np.tanh(context * 1.03)
    melody = np.tanh(melody * 1.01)

    melody_wav = args.out / "KRIS_EXACT_SANS_ASRIEL_MELODY.wav"
    context_wav = args.out / "KRIS_EXACT_SANS_ASRIEL_CONTEXT.wav"
    write_wav(melody_wav, melody)
    write_wav(context_wav, context)
    encode(melody_wav, args.out / "KRIS_EXACT_SANS_ASRIEL_MELODY.mp3")
    encode(context_wav, args.out / "KRIS_EXACT_SANS_ASRIEL_CONTEXT.mp3")
    export_midi(args.out / "KRIS_EXACT_SANS_ASRIEL.mid", events)
    (args.out / "KRIS_EXACT_SANS_ASRIEL_INSTRUMENTS.txt").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print("\n".join(report))


if __name__ == "__main__":
    main()
