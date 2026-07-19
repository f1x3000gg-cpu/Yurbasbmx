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
BAR_SECONDS = 4.0 * BEAT
BARS = 36
TAIL_SECONDS = 2.2
TPB = 480

PITCH = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

Event = Tuple[float, float, int, int]
Pattern = Sequence[Tuple[str | int | None, float]]

# Exact presets from popular UNDERTALE track groups. No saxophone preset exists
# in this renderer and no generic GM fallback is allowed.
PRESETS = {
    "Hopes Violin":     (2, 125, "087 Hopes and Dreams - Violin Detache"),
    "Asgore Piano":     (2,  64, "077 ASGORE - Piano"),
    "Asgore Timpani":   (2,  74, "077 ASGORE - Timpani"),
    "Asgore Strings":   (2,  75, "077 ASGORE - Strings"),
    "Asgore Choir":     (2,  76, "077 ASGORE - Choir Aahs"),
    "Spear Brass":      (1,  57, "046 Spear of Justice - Brass Trumpet"),
    "Hero Romantic":    (3,  65, "098 Battle Against a True Hero - Romantic Tp"),
    "Hero POWER":       (3,  66, "098 Battle Against a True Hero - POWER DrumKit"),
    "Finale Trumpet":   (2,  98, "080 Finale - Trumpet"),
    "Finale Strings":   (2,  97, "080 Finale - Strings"),
    "Neo Organ":        (3,  76, "099 Power of NEO - Organ 3"),
    "Core Saw":         (2,  38, "065 CORE - Saw"),
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
            add_note(events, track, cursor, nn(note) + shift, duration * .93, velocity)
        cursor += duration


def add_chord(events: Dict[str, List[Event]], track: str, start: float,
              notes: Sequence[str | int], duration: float, velocity: int) -> None:
    for note in notes:
        add_note(events, track, start, note, duration, velocity)


def make_events() -> Dict[str, List[Event]]:
    events: Dict[str, List[Event]] = {name: [] for name in PRESETS}

    # C minor. Written as a song-shaped melody before orchestration:
    # statement -> answer -> chorus -> bridge -> final chorus.
    theme_a: List[Pattern] = [
        [(None,.5),("G4",.5),("C5",1),("Eb5",1),("D5",1)],
        [("Bb4",.5),("C5",.5),("G4",1),("F4",1),("G4",1)],
        [(None,.5),("Ab4",.5),("C5",1),("F5",1),("Eb5",1)],
        [("D5",.5),("C5",.5),("B4",1),("G4",1),(None,1)],
        [("G4",.5),("C5",.5),("Eb5",1),("G5",1),("F5",1)],
        [("Eb5",.5),("D5",.5),("C5",1),("Bb4",1),("G4",1)],
        [("Ab4",.5),("C5",.5),("Eb5",1),("D5",.5),("C5",.5),("Bb4",1)],
        [("G4",.5),("B4",.5),("C5",2),(None,1)],
    ]

    answer_b: List[Pattern] = [
        [("C5",1),("G4",.5),("Ab4",.5),("Bb4",1),("Eb5",1)],
        [("D5",.5),("C5",.5),("Bb4",1),("G4",1),(None,1)],
        [("F4",.5),("Ab4",.5),("C5",1),("Eb5",1),("D5",1)],
        [("C5",.5),("Bb4",.5),("G4",1),("B4",1),("C5",1)],
        [("Eb5",1),("D5",.5),("C5",.5),("G5",1),("F5",1)],
        [("Eb5",.5),("C5",.5),("Bb4",1),("Ab4",1),("G4",1)],
        [("F4",.5),("Ab4",.5),("C5",1),("D5",.5),("Eb5",.5),("B4",1)],
        [("G4",.5),("B4",.5),("C5",2),(None,1)],
    ]

    chorus: List[Pattern] = [
        [("C5",1),("Eb5",.5),("G5",.5),("Ab5",1),("G5",1)],
        [("F5",.5),("Eb5",.5),("C5",1),("D5",1),("G5",1)],
        [("Ab5",1),("G5",.5),("Eb5",.5),("F5",1),("C5",1)],
        [("D5",.5),("Eb5",.5),("G5",1),("F5",.5),("D5",.5),("C5",1)],
        [("Eb5",.5),("G5",.5),("Bb5",1),("Ab5",.5),("G5",.5),("F5",1)],
        [("D5",.5),("Eb5",.5),("G5",1),("F5",.5),("Eb5",.5),("C5",1)],
        [("Ab4",.5),("C5",.5),("F5",1),("Eb5",.5),("D5",.5),("B4",1)],
        [("G4",.5),("B4",.5),("C5",2),(None,1)],
    ]

    bridge: List[Pattern] = [
        [("C5",2),("Bb4",1),("G4",1)],
        [("Ab4",1),("C5",1),("Eb5",2)],
        [("D5",1),("C5",.5),("Bb4",.5),("G4",2)],
        [("B4",1),("D5",1),("C5",2)],
    ]

    chords = [
        ["C3","G3","Bb3","D4"],
        ["Ab2","Eb3","G3","C4"],
        ["Eb3","G3","Bb3","D4"],
        ["G2","D3","F3","B3"],
        ["F3","C4","Eb4","Ab4"],
        ["Db3","Ab3","C4","F4"],
        ["G2","D3","F3","C4"],
        ["G2","D3","F3","B3"],
    ]
    roots = ["C2","Ab1","Eb2","G1","F2","Db2","G1","G1"]

    def add_harmony(bar: int, strong: bool) -> None:
        idx = bar % 8
        start = bar * 4
        add_chord(events, "Asgore Strings", start, chords[idx], 3.78, 48 if not strong else 62)
        if strong:
            add_chord(events, "Asgore Choir", start, chords[idx], 3.76, 34)
        add_note(events, "Core Saw", start, roots[idx], 1.65, 58 if not strong else 74)
        add_note(events, "Core Saw", start + 2, roots[idx], 1.65, 52 if not strong else 68)
        add_chord(events, "Asgore Piano", start, chords[idx], .34, 48 if not strong else 58)
        add_chord(events, "Asgore Piano", start + 2, chords[idx], .34, 44 if not strong else 54)

    def add_drums(bar: int, power: int, fill: bool = False) -> None:
        start = bar * 4
        for pos in (0, 2):
            add_note(events, "Hero POWER", start + pos, 36, .09, 94 + power * 6)
        for pos in (1, 3):
            add_note(events, "Hero POWER", start + pos, 38, .10, 104 + power * 5)
        for step in range(8):
            add_note(events, "Hero POWER", start + step * .5, 42, .05, 44 if step % 2 else 56)
        if power >= 1:
            add_note(events, "Hero POWER", start + 1.5, 36, .07, 72)
            add_note(events, "Hero POWER", start + 3.5, 36, .07, 76)
        if fill:
            for off, drum in [(3.0,45),(3.25,47),(3.5,48),(3.75,50)]:
                add_note(events, "Hero POWER", start + off, drum, .06, 78)

    # 0-7: serious first statement; melody remains clearly audible.
    for bar in range(8):
        start = bar * 4
        add_pattern(events, "Hopes Violin", start, theme_a[bar], 108)
        add_harmony(bar, False)
        add_drums(bar, 0, fill=(bar == 7))
        if bar in (0, 4):
            add_note(events, "Asgore Timpani", start, nn(roots[bar]) + 12, .55, 82)

    # 8-15: answer phrase, different notes and contour.
    for bar in range(8, 16):
        idx = bar - 8
        start = bar * 4
        add_pattern(events, "Spear Brass", start, answer_b[idx], 102)
        add_pattern(events, "Hopes Violin", start, answer_b[idx], 58, shift=-12)
        add_harmony(bar, True)
        add_drums(bar, 1, fill=(idx in (3, 7)))
        if idx in (0, 4):
            add_chord(events, "Neo Organ", start, [chords[idx][0],chords[idx][2]], 1.25, 62)

    # 16-23: memorable chorus; long notes carry the impact.
    for bar in range(16, 24):
        idx = bar - 16
        start = bar * 4
        add_pattern(events, "Finale Trumpet", start, chorus[idx], 112)
        add_pattern(events, "Hopes Violin", start, chorus[idx], 80)
        add_chord(events, "Finale Strings", start, chords[idx], 3.78, 60)
        add_chord(events, "Asgore Choir", start, chords[idx], 3.78, 40)
        add_chord(events, "Neo Organ", start, [chords[idx][0],chords[idx][2]], 3.72, 48)
        add_harmony(bar, True)
        add_drums(bar, 2, fill=(idx in (3, 7)))
        if idx in (0, 4):
            add_note(events, "Asgore Timpani", start, nn(roots[idx]) + 12, .62, 96)

    # 24-27: half-time bridge, no note spam.
    for bar in range(24, 28):
        idx = bar - 24
        start = bar * 4
        add_pattern(events, "Asgore Piano", start, bridge[idx], 102)
        add_pattern(events, "Hopes Violin", start, bridge[idx], 78)
        add_chord(events, "Asgore Strings", start, chords[idx + 4], 3.78, 58)
        add_chord(events, "Asgore Choir", start, chords[idx + 4], 3.78, 34)
        add_note(events, "Core Saw", start, roots[idx + 4], 3.6, 58)
        add_note(events, "Hero POWER", start, 36, .10, 96)
        add_note(events, "Hero POWER", start + 2, 38, .10, 108)
        if bar == 27:
            for off, drum in [(3.0,45),(3.25,47),(3.5,48),(3.75,50)]:
                add_note(events, "Hero POWER", start + off, drum, .06, 86)

    # 28-35: final chorus, stronger but still melody-first.
    for bar in range(28, 36):
        idx = bar - 28
        start = bar * 4
        add_pattern(events, "Finale Trumpet", start, chorus[idx], 118)
        add_pattern(events, "Hero Romantic", start, chorus[idx], 72, shift=-12)
        add_pattern(events, "Hopes Violin", start, chorus[idx], 92)
        add_chord(events, "Finale Strings", start, chords[idx], 3.78, 68)
        add_chord(events, "Asgore Choir", start, chords[idx], 3.78, 46)
        add_chord(events, "Neo Organ", start, [chords[idx][0],chords[idx][2]], 3.74, 56)
        add_harmony(bar, True)
        add_drums(bar, 2, fill=(idx in (3, 7)))
        if idx in (0, 4):
            add_note(events, "Asgore Timpani", start, nn(roots[idx]) + 12, .62, 104)

    # Final C-minor hit.
    end = BARS * 4 - .5
    final = ["C3","G3","C4","Eb4","G4"]
    add_chord(events, "Finale Trumpet", end, final, .48, 120)
    add_chord(events, "Finale Strings", end, final, .58, 116)
    add_chord(events, "Neo Organ", end, final, .58, 110)
    add_note(events, "Hero POWER", end, 49, .28, 124)
    add_note(events, "Asgore Timpani", end, "C3", .60, 120)

    return events


def render_stem(soundfont: Path, bank: int, preset: int,
                events: Sequence[Event], gain: float = .78) -> np.ndarray:
    total = int((BARS * BAR_SECONDS + TAIL_SECONDS) * SR)
    synth = fluidsynth.Synth(gain=gain, samplerate=SR)
    sfid = synth.sfload(str(soundfont))
    result = synth.program_select(0, sfid, bank, preset)
    if result != 0:
        synth.delete()
        raise RuntimeError(f"Failed exact preset bank={bank} preset={preset}")
    synth.set_reverb(roomsize=.17, damping=.67, width=.72, level=.09)
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
        "acompressor=threshold=-17dB:ratio=2.0:attack=10:release=120,"
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
    total = int((BARS * BAR_SECONDS + TAIL_SECONDS) * SR)
    full = np.zeros((total, 2), dtype=np.float64)
    proof = np.zeros_like(full)

    gains = {
        "Hopes Violin": .78,
        "Asgore Piano": .48,
        "Asgore Timpani": .42,
        "Asgore Strings": .50,
        "Asgore Choir": .30,
        "Spear Brass": .68,
        "Hero Romantic": .40,
        "Hero POWER": .68,
        "Finale Trumpet": .82,
        "Finale Strings": .52,
        "Neo Organ": .42,
        "Core Saw": .34,
    }
    proof_tracks = {
        "Hopes Violin": .90,
        "Asgore Piano": .52,
        "Asgore Strings": .44,
        "Spear Brass": .70,
    }

    lines = [
        f"BPM: {BPM}",
        "Key: C minor",
        "No saxophone. No random-note generator. No generic GM fallback.",
        "Structure: theme A -> answer B -> chorus -> bridge -> final chorus.",
        "",
    ]

    for name, track_events in events.items():
        if not track_events:
            continue
        bank, preset, source = PRESETS[name]
        lines.append(f"{name}: bank={bank} preset={preset} | {source}")
        stem = render_stem(args.soundfont, bank, preset, track_events)
        length = min(total, len(stem))
        full[:length] += stem[:length] * gains[name]
        if name in proof_tracks:
            proof_len = min(length, int((16 * BAR_SECONDS + 1.0) * SR))
            proof[:proof_len] += stem[:proof_len] * proof_tracks[name]

    full = np.tanh(full * 1.05)
    proof = np.tanh(proof * 1.02)
    proof = proof[:int((16 * BAR_SECONDS + 1.2) * SR)]

    full_wav = args.out / "KRIS_SERIOUS_NO_SAX_FULL.wav"
    proof_wav = args.out / "KRIS_SERIOUS_NO_SAX_MELODY.wav"
    write_wav(full_wav, full)
    write_wav(proof_wav, proof)
    encode(full_wav,
           args.out / "KRIS_SERIOUS_NO_SAX_FULL.mp3",
           args.out / "KRIS_SERIOUS_NO_SAX_GAME.ogg")
    encode(proof_wav, args.out / "KRIS_SERIOUS_NO_SAX_MELODY.mp3")
    export_midi(args.out / "KRIS_SERIOUS_NO_SAX.mid", events)
    (args.out / "KRIS_SERIOUS_NO_SAX_INSTRUMENTS.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
