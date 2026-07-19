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
BAR_SECONDS = 4.0 * BEAT
BARS = 32
TAIL_SECONDS = 2.5
TPB = 480

PITCH = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

Event = Tuple[float, float, int, int]
Pattern = Sequence[Tuple[str | int | None, float]]

# Exact named track-group presets from UNDERTALE Soundfont 2.
# No saxophone and no generic GM fallback are used.
PRESETS = {
    "Asgore Piano":   (2, 64, "077 ASGORE - Piano"),
    "Asgore Violin":  (2, 73, "077 ASGORE - Violin"),
    "Asgore Timpani": (2, 74, "077 ASGORE - Timpani"),
    "Asgore Strings": (2, 75, "077 ASGORE - Strings"),
    "Asgore Choir":   (2, 76, "077 ASGORE - Choir Aahs"),
    "Asgore Hit":     (2, 77, "077 ASGORE - Orchestra Hit"),
    "Asgore Brass":   (2, 78, "077 ASGORE - Brass"),
    "Asgore Drums":   (2, 80, "077 ASGORE - Drumkit"),
    "Finale Trumpet": (2, 98, "080 Finale - Trumpet"),
    "Finale Strings": (2, 99, "080 Finale - Strings marc"),
    "Core Brass":     (2, 35, "065 CORE - Brass"),
    "Core Piano":     (2, 37, "065 CORE - Piano"),
    "Neo Organ":      (3, 76, "099 Power of NEO - Organ 3"),
    "Hero Tubular":   (3, 71, "098 Battle Against a True Hero - Tubular Bells"),
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

    # C minor. Written as a royal command, an answer, and a larger final decree.
    # No random generator and no stream of melodic sixteenth notes.
    theme_a: List[Pattern] = [
        [(None,.5),("G4",.5),("C5",1),("Eb5",.5),("D5",.5),("C5",1)],
        [("Bb4",.5),("G4",.5),("F4",1),("Ab4",1),("G4",1)],
        [(None,.5),("C5",.5),("F5",1),("Eb5",1),("D5",1)],
        [("B4",.5),("C5",.5),("D5",1),("G4",1),("C5",1)],
        [("C5",1),("Eb5",.5),("G5",.5),("F5",1),("Eb5",1)],
        [("D5",.5),("C5",.5),("Bb4",1),("Ab4",1),("G4",1)],
        [("Ab4",.5),("C5",.5),("Eb5",1),("D5",.5),("C5",.5),("G4",1)],
        [("B4",.5),("D5",.5),("F5",1),("Eb5",.5),("D5",.5),("C5",1)],
    ]

    answer_b: List[Pattern] = [
        [("C5",1),("G4",.5),("Ab4",.5),("Bb4",1),("Eb5",1)],
        [("D5",.5),("C5",.5),("Bb4",1),("G4",2)],
        [("F4",.5),("Ab4",.5),("C5",1),("Eb5",.5),("D5",.5),("C5",1)],
        [("B4",.5),("C5",.5),("G5",1),("F5",1),("D5",1)],
        [("Eb5",1),("D5",.5),("C5",.5),("G4",1),("Bb4",1)],
        [("Ab4",.5),("Bb4",.5),("C5",1),("D5",.5),("Eb5",.5),("F5",1)],
        [("G5",1),("F5",.5),("Eb5",.5),("D5",1),("Bb4",1)],
        [("B4",.5),("D5",.5),("G5",1),("F5",.5),("D5",.5),("C5",1)],
    ]

    chorus: List[Pattern] = [
        [("G5",1),("Eb5",.5),("F5",.5),("G5",1),("Bb5",1)],
        [("Ab5",1),("G5",.5),("F5",.5),("Eb5",2)],
        [("C5",.5),("Eb5",.5),("G5",1),("C6",1),("Bb5",1)],
        [("Ab5",.5),("G5",.5),("F5",1),("D5",1),("G5",1)],
        [("Eb5",1),("G5",.5),("Bb5",.5),("Ab5",1),("G5",1)],
        [("F5",.5),("Eb5",.5),("D5",1),("C5",2)],
        [("Ab4",.5),("C5",.5),("Eb5",1),("G5",.5),("F5",.5),("D5",1)],
        [("B4",.5),("D5",.5),("G5",1),("F5",.5),("D5",.5),("C5",1)],
    ]

    chords = [
        ["C3","G3","D4","Eb4"],
        ["Ab2","Eb3","G3","C4"],
        ["F3","C4","Eb4","Ab4"],
        ["G2","D3","F3","B3"],
        ["Eb3","G3","C4","Eb4"],
        ["Ab2","Eb3","G3","C4"],
        ["D3","Ab3","C4","F4"],
        ["G2","D3","F3","B3"],
    ]
    roots = ["C2","Ab1","F2","G1","Eb2","Ab1","D2","G1"]

    def add_harmony_bar(bar: int, intensity: int) -> None:
        start = bar * 4
        idx = bar % 8
        add_chord(events, "Neo Organ", start, chords[idx], 3.78, 46 + intensity * 5)
        add_chord(events, "Asgore Strings", start, chords[idx], 3.78, 40 + intensity * 6)
        if intensity >= 1:
            add_chord(events, "Asgore Choir", start, chords[idx], 3.78, 25 + intensity * 5)
        add_note(events, "Core Piano", start, roots[idx], 1.7, 64 + intensity * 7)
        add_note(events, "Core Piano", start + 2, roots[idx], 1.7, 58 + intensity * 6)

    def add_martial_drums(bar: int, intensity: int, fill: bool = False) -> None:
        start = bar * 4
        add_note(events, "Asgore Drums", start, 36, .10, 92 + intensity * 6)
        add_note(events, "Asgore Drums", start + 2, 36, .10, 88 + intensity * 6)
        add_note(events, "Asgore Drums", start + 1, 38, .11, 100 + intensity * 5)
        add_note(events, "Asgore Drums", start + 3, 38, .11, 106 + intensity * 5)
        for step in range(8):
            add_note(events, "Asgore Drums", start + step * .5, 42, .055,
                     38 + (8 if step % 2 == 0 else 0) + intensity * 3)
        if intensity >= 2:
            add_note(events, "Asgore Drums", start + 1.5, 36, .07, 74)
            add_note(events, "Asgore Drums", start + 3.5, 36, .07, 80)
        if fill:
            for offset, drum_note in [(3.0,45),(3.25,47),(3.5,48),(3.75,50)]:
                add_note(events, "Asgore Drums", start + offset, drum_note, .07, 82 + intensity * 4)

    def add_timpani_mark(bar: int, strong: bool = False) -> None:
        start = bar * 4
        idx = bar % 8
        pitch = note_number(roots[idx]) + 12
        add_note(events, "Asgore Timpani", start, pitch, .55, 82 if not strong else 106)
        if strong:
            add_note(events, "Asgore Timpani", start + 2, pitch + 7, .42, 88)

    # 0-3: the king enters. No comedy and no saxophone.
    for bar in range(4):
        idx = bar
        start = bar * 4
        add_pattern(events, "Asgore Brass", start, theme_a[idx], 92)
        add_pattern(events, "Asgore Piano", start, theme_a[idx], 58, shift=-12)
        add_harmony_bar(bar, 0)
        add_timpani_mark(bar, strong=(bar in (0,3)))
        if bar in (0,3):
            add_chord(events, "Asgore Hit", start, [chords[idx][0],chords[idx][2]], .24, 92)

    # 4-11: complete royal theme.
    for bar in range(4, 12):
        idx = bar - 4
        start = bar * 4
        add_pattern(events, "Finale Trumpet", start, theme_a[idx], 110)
        add_pattern(events, "Asgore Violin", start, theme_a[idx], 64, shift=-12)
        add_harmony_bar(bar, 1)
        add_martial_drums(bar, 1, fill=(idx % 4 == 3))
        add_timpani_mark(bar, strong=(idx in (0,4)))
        if idx in (0,4):
            add_chord(events, "Asgore Hit", start,
                      [chords[idx][0],chords[idx][2],chords[idx][3]], .24, 104)

    # 12-19: answer phrase, colder and heavier.
    for bar in range(12, 20):
        idx = bar - 12
        start = bar * 4
        add_pattern(events, "Core Brass", start, answer_b[idx], 108)
        add_pattern(events, "Asgore Piano", start, answer_b[idx], 68, shift=-12)
        add_harmony_bar(bar, 1)
        add_martial_drums(bar, 1, fill=(idx % 4 == 3))
        if idx in (3,7):
            add_note(events, "Hero Tubular", start + 3.5, "C5" if idx == 3 else "C6", .34, 72)

    # 20-27: final decree. Bigger melody, still readable and singable.
    for bar in range(20, 28):
        idx = bar - 20
        start = bar * 4
        add_pattern(events, "Finale Trumpet", start, chorus[idx], 116)
        add_pattern(events, "Finale Strings", start, chorus[idx], 78, shift=-12)
        add_pattern(events, "Asgore Brass", start, chorus[idx], 62, shift=-12)
        add_harmony_bar(bar, 2)
        add_martial_drums(bar, 2, fill=(idx % 2 == 1))
        add_timpani_mark(bar, strong=(idx in (0,4)))
        if idx in (0,4):
            add_chord(events, "Asgore Hit", start,
                      [chords[idx][0],chords[idx][2],chords[idx][3]], .26, 116)

    # 28-31: transformation/coda, a compressed return of the first statement.
    for bar in range(28, 32):
        idx = bar - 28
        start = bar * 4
        add_pattern(events, "Finale Trumpet", start, theme_a[idx + 4], 120)
        add_pattern(events, "Asgore Violin", start, theme_a[idx + 4], 86)
        add_pattern(events, "Core Brass", start, theme_a[idx + 4], 58, shift=-12)
        add_harmony_bar(bar, 2)
        add_martial_drums(bar, 2, fill=True)
        add_timpani_mark(bar, strong=True)

    final_start = BARS * 4 - .5
    final_chord = ["C2","G2","C3","Eb3","G3","C4"]
    add_chord(events, "Asgore Hit", final_start, final_chord, .42, 127)
    add_chord(events, "Neo Organ", final_start, final_chord, .72, 118)
    add_chord(events, "Asgore Brass", final_start, final_chord, .58, 122)
    add_note(events, "Asgore Drums", final_start, 49, .32, 127)
    add_note(events, "Asgore Timpani", final_start, "C3", .62, 127)

    return events


def render_stem(soundfont: Path, bank: int, preset: int,
                events: Sequence[Event], gain: float = .78) -> np.ndarray:
    total = int((BARS * BAR_SECONDS + TAIL_SECONDS) * SR)
    synth = fluidsynth.Synth(gain=gain, samplerate=SR)
    sfid = synth.sfload(str(soundfont))
    result = synth.program_select(0, sfid, bank, preset)
    if result != 0:
        synth.delete()
        raise RuntimeError(f"Failed to select exact bank={bank} preset={preset}")
    synth.set_reverb(roomsize=.20, damping=.68, width=.78, level=.12)
    synth.set_chorus(nr=2, level=.12, speed=.23, depth=1.7, type=0)

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
        "highpass=f=26,lowpass=f=12000,"
        "acompressor=threshold=-18dB:ratio=2.2:attack=10:release=130,"
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

    events = build_events()
    total = int((BARS * BAR_SECONDS + TAIL_SECONDS) * SR)
    full = np.zeros((total, 2), dtype=np.float64)
    proof = np.zeros_like(full)

    gains = {
        "Asgore Piano": .50,
        "Asgore Violin": .52,
        "Asgore Timpani": .55,
        "Asgore Strings": .58,
        "Asgore Choir": .34,
        "Asgore Hit": .58,
        "Asgore Brass": .68,
        "Asgore Drums": .70,
        "Finale Trumpet": .82,
        "Finale Strings": .56,
        "Core Brass": .56,
        "Core Piano": .42,
        "Neo Organ": .52,
        "Hero Tubular": .30,
    }

    proof_tracks = {
        "Finale Trumpet": .90,
        "Asgore Brass": .78,
        "Asgore Violin": .52,
        "Asgore Piano": .44,
        "Asgore Strings": .40,
        "Neo Organ": .32,
    }

    selected_lines = [
        f"BPM: {BPM}",
        "Serious king theme. No saxophone. No generic GM fallback.",
        "Original melody: royal command -> answer -> final decree.",
        "",
    ]

    for name, track_events in events.items():
        if not track_events:
            continue
        bank, preset, source = PRESETS[name]
        selected_lines.append(f"{name}: bank={bank} preset={preset} | {source}")
        stem = render_stem(args.soundfont, bank, preset, track_events)
        length = min(total, len(stem))
        full[:length] += stem[:length] * gains.get(name, 0.0)
        if name in proof_tracks:
            proof_length = min(length, int((20 * BAR_SECONDS + .9) * SR))
            proof[:proof_length] += stem[:proof_length] * proof_tracks[name]

    full = np.tanh(full * 1.05)
    proof = np.tanh(proof * 1.02)
    proof = proof[:int((20 * BAR_SECONDS + 1.2) * SR)]

    full_wav = args.out / "KING_SERIOUS_THEME_FULL.wav"
    proof_wav = args.out / "KING_SERIOUS_THEME_MELODY.wav"
    write_wav(full_wav, full)
    write_wav(proof_wav, proof)

    encode(full_wav,
           args.out / "KING_SERIOUS_THEME_FULL.mp3",
           args.out / "KING_SERIOUS_THEME_GAME.ogg")
    encode(proof_wav, args.out / "KING_SERIOUS_THEME_MELODY.mp3")
    export_midi(args.out / "KING_SERIOUS_THEME.mid", events)
    (args.out / "KING_SERIOUS_THEME_INSTRUMENTS.txt").write_text(
        "\n".join(selected_lines) + "\n", encoding="utf-8"
    )

    print("\n".join(selected_lines))


if __name__ == "__main__":
    main()
