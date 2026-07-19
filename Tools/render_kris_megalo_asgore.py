#!/usr/bin/env python3
from __future__ import annotations

import argparse
import struct
import subprocess
import wave
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import fluidsynth
import numpy as np
from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo

SR = 44100
BPM = 148
BEAT = 60.0 / BPM
BARS = 16
TAIL = 2.5
TPB = 480

PITCH = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

Event = Tuple[float, float, int, int]
Pattern = Sequence[Tuple[Optional[str], float]]
Preset = Tuple[int, int, str]


def nn(note: str) -> int:
    if len(note) >= 3 and note[1] in "#b":
        name, octave = note[:2], int(note[2:])
    else:
        name, octave = note[0], int(note[1:])
    return 12 * (octave + 1) + PITCH[name]


def parse_presets(path: Path) -> List[Preset]:
    data = path.read_bytes()
    marker = data.find(b"phdr")
    if marker < 0:
        return []
    size = struct.unpack_from("<I", data, marker + 4)[0]
    chunk = data[marker + 8: marker + 8 + size]
    presets: List[Preset] = []
    for offset in range(0, max(0, len(chunk) - 38), 38):
        record = chunk[offset: offset + 38]
        name = record[:20].split(b"\0", 1)[0].decode("latin-1", errors="replace").strip()
        preset = struct.unpack_from("<H", record, 20)[0]
        bank = struct.unpack_from("<H", record, 22)[0]
        presets.append((bank, preset, name))
    return presets


def optional_asgore_orchestra_hit(soundfont: Path) -> Optional[Preset]:
    candidates = []
    for bank, preset, name in parse_presets(soundfont):
        text = name.lower()
        if bank == 2 and 64 <= preset <= 80 and "orchestra" in text:
            candidates.append((bank, preset, name))
    return sorted(candidates, key=lambda item: item[1])[0] if candidates else None


def add(events: Dict[str, List[Event]], track: str, start: float,
        note: str | int, duration: float, velocity: int) -> None:
    pitch = note if isinstance(note, int) else nn(note)
    events[track].append((float(start), float(duration), int(pitch), int(velocity)))


def phrase(events: Dict[str, List[Event]], track: str, start: float,
           notes: Pattern, velocity: int, gate: float = 0.88,
           accents: Sequence[int] = ()) -> None:
    cursor = start
    for index, (note, duration) in enumerate(notes):
        if note is not None:
            dynamic = min(127, velocity + (8 if index in accents else 0))
            add(events, track, cursor, note, max(0.06, duration * gate), dynamic)
        cursor += duration


def chord(events: Dict[str, List[Event]], track: str, start: float,
          notes: Sequence[str], duration: float, velocity: int) -> None:
    for note in notes:
        add(events, track, start, note, duration, velocity)


def lead_melody() -> List[Pattern]:
    # Hand-written 16-bar melody: held notes, quick pickups, rests and a real climax.
    return [
        [("D4", .5), ("F4", .5), ("A4", 1.0), ("G4", .5), ("F4", .5), ("E4", 1.0)],
        [("F4", .5), ("G4", .5), ("A4", .5), ("C5", .5), ("A4", 1.0), ("G4", .5), ("F4", .5)],
        [("E4", .75), ("F4", .25), ("G4", .5), ("A4", .5), ("C5", 1.5), ("A4", .5)],
        [("G4", .25), ("A4", .25), ("Bb4", .25), ("A4", .25), ("G4", .5), ("F4", .5), ("E4", 1.0), ("D4", 1.0)],
        [(None, .5), ("D4", .5), ("F4", .5), ("A4", 1.5), ("G4", .5), ("F4", .5)],
        [("E4", .5), ("G4", .5), ("A4", 1.0), ("C5", .5), ("Bb4", .5), ("A4", 1.0)],
        [("F4", .25), ("G4", .25), ("A4", .25), ("Bb4", .25), ("C5", .5), ("D5", .5), ("C5", .5), ("Bb4", .5), ("A4", 1.0)],
        [("G4", .5), ("F4", .5), ("E4", .5), ("D4", .5), ("C#4", .5), ("E4", .5), ("A4", 1.0)],
        [("A4", 1.5), ("C5", .5), ("D5", .5), ("C5", .5), ("A4", 1.0)],
        [("F4", .5), ("A4", .5), ("C5", .75), ("Bb4", .25), ("A4", .5), ("G4", .5), ("F4", 1.0)],
        [("E4", .25), ("F4", .25), ("G4", .25), ("A4", .25), ("C5", .5), ("D5", .5), ("C5", 1.0), ("A4", 1.0)],
        [("C5", .5), ("A4", .5), ("G4", .5), ("F4", .5), ("E4", 1.5), ("D4", .5)],
        [("D4", .5), ("F4", .5), ("A4", 1.0), ("C5", 1.0), ("D5", 1.0)],
        [("C5", .25), ("Bb4", .25), ("A4", .5), ("G4", .5), ("F4", .5), ("A4", 1.0), ("C5", 1.0)],
        [("D5", 1.5), ("C5", .5), ("A4", .5), ("G4", .5), ("F4", .5), ("E4", .5)],
        [("F4", .25), ("G4", .25), ("A4", .25), ("C5", .25), ("Bb4", .5), ("A4", .5), ("G4", .5), ("E4", .5), ("D4", 1.0)],
    ]


def build_events(use_orchestra_hit: bool) -> Dict[str, List[Event]]:
    names = [
        "MEGALOVANIA Lead Guitar",
        "MEGALOVANIA Background Guitar",
        "MEGALOVANIA Bass",
        "ASGORE Timpani",
        "ASGORE Drums",
    ]
    if use_orchestra_hit:
        names.append("ASGORE Orchestra Hit")
    events = {name: [] for name in names}

    melody = lead_melody()
    progression = [
        ("D2", ["D3", "A3", "D4"]),
        ("Bb1", ["Bb2", "F3", "Bb3"]),
        ("F2", ["F3", "C4", "F4"]),
        ("C2", ["C3", "G3", "C4"]),
        ("D2", ["D3", "A3", "D4"]),
        ("Bb1", ["Bb2", "F3", "Bb3"]),
        ("G1", ["G2", "D3", "G3"]),
        ("A1", ["A2", "E3", "A3"]),
    ] * 2

    for bar in range(BARS):
        start = bar * 4.0
        section_velocity = 76 if bar < 4 else 80 if bar < 8 else 86 if bar < 12 else 92
        phrase(
            events,
            "MEGALOVANIA Lead Guitar",
            start,
            melody[bar],
            section_velocity,
            gate=0.93 if bar in (2, 4, 8, 12, 14) else 0.84,
            accents=(0, 3),
        )

        root, power = progression[bar]

        # Background guitar is rhythm in the verse, then a real harmony in the second half.
        if bar < 8:
            chord(events, "MEGALOVANIA Background Guitar", start, power, 1.35, 43)
            chord(events, "MEGALOVANIA Background Guitar", start + 2.0, power, 1.25, 38)
        else:
            chord(events, "MEGALOVANIA Background Guitar", start, power, 1.75, 48)
            chord(events, "MEGALOVANIA Background Guitar", start + 2.0, power, 1.65, 45)
            # Small counter-line binds the harmony to the lead instead of sounding separate.
            counter = [
                [("F4", .5), ("E4", .5), ("D4", 1.0), (None, 2.0)],
                [("D4", .5), ("F4", .5), ("G4", 1.0), (None, 2.0)],
                [("A3", .5), ("C4", .5), ("F4", 1.0), (None, 2.0)],
                [("E4", .5), ("D4", .5), ("C4", 1.0), (None, 2.0)],
            ][bar % 4]
            phrase(events, "MEGALOVANIA Background Guitar", start + 1.0,
                   counter, 42, gate=0.78)

        # Bass follows the harmonic motion and uses passing notes before cadences.
        r = nn(root)
        add(events, "MEGALOVANIA Bass", start, r, .72, 58)
        add(events, "MEGALOVANIA Bass", start + 1.0, r + 7, .55, 46)
        add(events, "MEGALOVANIA Bass", start + 2.0, r + 12, .65, 54)
        add(events, "MEGALOVANIA Bass", start + 3.0, r + (10 if bar in (3, 7, 11, 15) else 7), .50, 45)

        # The approved battle groove, with denser fills only at phrase boundaries.
        for pos, vel in ((0.0, 100), (2.0, 94)):
            add(events, "ASGORE Drums", start + pos, 36, .07, vel)
        for pos, vel in ((1.0, 110), (3.0, 106)):
            add(events, "ASGORE Drums", start + pos, 38, .08, vel)
        for step in range(8):
            add(events, "ASGORE Drums", start + step * .5, 42, .04,
                46 if step % 2 else 56)
        if bar in (3, 7, 11, 15):
            fill = [(3.0, 45), (3.25, 47), (3.5, 48), (3.75, 50)]
            for off, drum in fill:
                add(events, "ASGORE Drums", start + off, drum, .05, 78)
        if bar >= 8 and bar % 2 == 1:
            add(events, "ASGORE Drums", start + 1.75, 36, .05, 76)

        # ASGORE color is structural: timpani and optional orchestra hits glue sections together.
        if bar in (0, 4, 8, 12):
            add(events, "ASGORE Timpani", start, r + 12, .65, 78)
        if bar in (3, 7, 11, 15):
            add(events, "ASGORE Timpani", start + 3.0, r + 12, .80, 86)
        if use_orchestra_hit and bar in (4, 8, 12, 15):
            add(events, "ASGORE Orchestra Hit", start if bar != 15 else start + 3.0,
                nn("D4") if bar != 15 else nn("A3"), .50, 62)

    return events


def render_stem(soundfont: Path, preset: Preset, events: Sequence[Event]) -> np.ndarray:
    total = int(((BARS * 4 * BEAT) + TAIL) * SR)
    bank, program, _ = preset
    synth = fluidsynth.Synth(gain=.72, samplerate=SR)
    sfid = synth.sfload(str(soundfont))
    if synth.program_select(0, sfid, bank, program) != 0:
        synth.delete()
        raise RuntimeError(f"Cannot select preset bank={bank} program={program}")
    synth.set_reverb(roomsize=.13, damping=.72, width=.72, level=.055)
    synth.set_chorus(nr=2, level=.035, speed=.22, depth=.90, type=0)

    timeline = []
    for start, duration, note, velocity in events:
        timeline.append((int(start * BEAT * SR), 1, note, velocity))
        timeline.append((int((start + duration) * BEAT * SR), 0, note, 0))
    timeline.sort(key=lambda item: (item[0], item[1]))

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
    if cursor < total:
        chunks.append(synth.get_samples(total - cursor))
    synth.delete()

    if not chunks:
        return np.zeros((total, 2), dtype=np.float64)
    raw = np.concatenate(chunks).astype(np.float64) / 32768.0
    return raw.reshape(-1, 2)


def pan(audio: np.ndarray, amount: float) -> np.ndarray:
    # amount: -1 left, +1 right
    left = np.sqrt((1.0 - amount) * 0.5)
    right = np.sqrt((1.0 + amount) * 0.5)
    result = audio.copy()
    result[:, 0] *= left
    result[:, 1] *= right
    return result


def mix(soundfont: Path, presets: Dict[str, Preset], events: Dict[str, List[Event]]) -> np.ndarray:
    total = int(((BARS * 4 * BEAT) + TAIL) * SR)
    output = np.zeros((total, 2), dtype=np.float64)
    gains = {
        "MEGALOVANIA Lead Guitar": .58,
        "MEGALOVANIA Background Guitar": .33,
        "MEGALOVANIA Bass": .30,
        "ASGORE Timpani": .36,
        "ASGORE Drums": .58,
        "ASGORE Orchestra Hit": .26,
    }
    pans = {
        "MEGALOVANIA Lead Guitar": 0.00,
        "MEGALOVANIA Background Guitar": -0.24,
        "MEGALOVANIA Bass": 0.00,
        "ASGORE Timpani": 0.10,
        "ASGORE Drums": 0.00,
        "ASGORE Orchestra Hit": 0.22,
    }

    for name, track_events in events.items():
        if not track_events:
            continue
        stem = render_stem(soundfont, presets[name], track_events)
        stem = pan(stem, pans.get(name, 0.0))
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


def encode(wav_path: Path, mp3_path: Path, ogg_path: Path) -> None:
    audio_filter = (
        "highpass=f=48,lowpass=f=12800,"
        "acompressor=threshold=-19dB:ratio=1.55:attack=18:release=145,"
        "alimiter=limit=.95"
    )
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path),
        "-af", audio_filter, "-codec:a", "libmp3lame", "-q:a", "2", str(mp3_path),
    ], check=True)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path),
        "-af", audio_filter, "-codec:a", "libvorbis", "-q:a", "6", str(ogg_path),
    ], check=True)


def export_midi(path: Path, presets: Dict[str, Preset], events: Dict[str, List[Event]]) -> None:
    midi = MidiFile(ticks_per_beat=TPB)
    tempo = MidiTrack()
    tempo.append(MetaMessage("set_tempo", tempo=bpm2tempo(BPM), time=0))
    tempo.append(MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    midi.tracks.append(tempo)

    channel = 0
    for name, track_events in events.items():
        if not track_events:
            continue
        while channel == 9:
            channel += 1
        ch = channel % 16
        channel += 1
        bank, program, _ = presets[name]
        track = MidiTrack()
        track.append(MetaMessage("track_name", name=name, time=0))
        track.append(Message("control_change", channel=ch, control=0, value=bank, time=0))
        track.append(Message("program_change", channel=ch, program=program, time=0))
        timeline = []
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--soundfont", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    orchestra_hit = optional_asgore_orchestra_hit(args.soundfont)
    presets: Dict[str, Preset] = {
        "MEGALOVANIA Lead Guitar": (0, 0, "100 MEGALOVANIA - Overdriven Guitar"),
        "MEGALOVANIA Background Guitar": (0, 13, "100 MEGALOVANIA - Background Guitar"),
        "MEGALOVANIA Bass": (0, 14, "100 MEGALOVANIA - Bass"),
        "ASGORE Timpani": (2, 74, "077 ASGORE - Timpani"),
        "ASGORE Drums": (2, 80, "077 ASGORE - Drumkit"),
    }
    if orchestra_hit is not None:
        presets["ASGORE Orchestra Hit"] = orchestra_hit

    events = build_events(orchestra_hit is not None)
    audio = mix(args.soundfont, presets, events)

    wav_path = args.out / "KRIS_REAL_MELODIC_BATTLE_FULL.wav"
    mp3_path = args.out / "KRIS_REAL_MELODIC_BATTLE_FULL.mp3"
    ogg_path = args.out / "KRIS_REAL_MELODIC_BATTLE_GAME.ogg"
    midi_path = args.out / "KRIS_REAL_MELODIC_BATTLE.mid"
    info_path = args.out / "KRIS_REAL_MELODIC_BATTLE_INSTRUMENTS.txt"

    write_wav(wav_path, audio)
    encode(wav_path, mp3_path, ogg_path)
    export_midi(midi_path, presets, events)

    lines = [
        f"BPM: {BPM}",
        "Key center: D minor",
        "Length: 16 bars",
        "Structure: hook / answer / chorus / climax",
        "Main melody: middle-register MEGALOVANIA Overdriven Guitar",
        "Harmony: MEGALOVANIA Background Guitar",
        "Moving bass: MEGALOVANIA Bass",
        "Rhythm and transitions: ASGORE Drumkit + Timpani",
        f"ASGORE Orchestra Hit used: {'yes — ' + orchestra_hit[2] if orchestra_hit else 'no (preset not present in compiled bank)'}",
        "No saxophone, clarinet, flute, trumpet lead, violin lead, pulse lead or piano lead.",
        "The melody contains held notes, fast pickups, rests, dynamic accents and a final climax.",
    ]
    info_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
