#!/usr/bin/env python3
from __future__ import annotations

import argparse
import struct
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import fluidsynth
import numpy as np
from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo

SR = 44100
BPM = 144
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
Pattern = Sequence[Tuple[str | None, float]]
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
        raise RuntimeError("SoundFont phdr chunk not found")
    size = struct.unpack_from("<I", data, marker + 4)[0]
    chunk = data[marker + 8:marker + 8 + size]
    result: List[Preset] = []
    for offset in range(0, len(chunk) - 38, 38):
        record = chunk[offset:offset + 38]
        name = record[:20].split(b"\0", 1)[0].decode("latin-1", errors="replace").strip()
        preset = struct.unpack_from("<H", record, 20)[0]
        bank = struct.unpack_from("<H", record, 22)[0]
        result.append((bank, preset, name))
    return result


def find_in_asgore_group(presets: Sequence[Preset], tokens: Sequence[str], exclude: Sequence[str] = ()) -> Preset:
    # The compiled bank's ASGORE group is bank 2, presets 64..80.
    matches: List[Preset] = []
    for item in presets:
        bank, preset, name = item
        text = name.lower()
        if bank != 2 or not 64 <= preset <= 80:
            continue
        if all(token.lower() in text for token in tokens) and not any(token.lower() in text for token in exclude):
            matches.append(item)
    if not matches:
        group = "\n".join(f"{b}:{p} {n}" for b, p, n in presets if b == 2 and 64 <= p <= 80)
        raise RuntimeError(f"ASGORE preset not found for {tokens}; group contents:\n{group}")
    return sorted(matches, key=lambda item: item[1])[0]


def resolve_presets(soundfont: Path) -> Dict[str, Preset]:
    available = parse_presets(soundfont)
    return {
        "ASGORE Pulse 25": find_in_asgore_group(available, ("25", "pulse"), ("12.5",)),
        "ASGORE Pulse 12.5": find_in_asgore_group(available, ("12", "pulse")),
        # These mappings were already verified by successful earlier renders.
        "MEGALOVANIA Soft Guitar": (0, 0, "100 MEGALOVANIA - Overdriven Guitar"),
        "ASGORE Drums": (2, 80, "077 ASGORE - Drumkit"),
    }


def add(events: Dict[str, List[Event]], track: str, start: float,
        note: str | int, duration: float, velocity: int) -> None:
    events[track].append((start, duration, note if isinstance(note, int) else nn(note), velocity))


def pattern(events: Dict[str, List[Event]], track: str, start: float,
            notes: Pattern, velocity: int, gate: float) -> None:
    cursor = start
    for note, duration in notes:
        if note is not None:
            add(events, track, cursor, note, max(.05, duration * gate), velocity)
        cursor += duration


def melody() -> List[Pattern]:
    return [
        [("D4", .5), ("F4", .5), ("A4", 1), ("G4", .5), ("F4", .5), ("D4", 1)],
        [("C4", .5), ("E4", .5), ("G4", 1), ("A4", .5), ("G4", .5), ("E4", 1)],
        [("Bb3", .5), ("D4", .5), ("F4", 1), ("A4", .5), ("G4", .5), ("F4", 1)],
        [("C#4", .5), ("E4", .5), ("A4", 1), ("G4", .5), ("E4", .5), ("D4", 1)],
        [("F4", .5), ("A4", .5), ("C5", 1), ("A4", .5), ("G4", .5), ("F4", 1)],
        [("E4", .5), ("G4", .5), ("Bb4", 1), ("A4", .5), ("G4", .5), ("E4", 1)],
        [("D4", .5), ("F4", .5), ("A4", .5), ("C5", .5), ("Bb4", .5), ("A4", .5), ("F4", 1)],
        [("E4", .5), ("G4", .5), ("A4", 1), ("C#5", .5), ("A4", .5), ("D4", 1)],
    ]


def make_events(names: Sequence[str]):
    pulse = {name: [] for name in names}
    guitar = {name: [] for name in names}
    context = {name: [] for name in names}
    hook = melody()

    for bar in range(4):
        start = bar * 4.0
        pattern(pulse, "ASGORE Pulse 25", start, hook[bar], 88, .82)
        pattern(pulse, "ASGORE Pulse 12.5", start, hook[bar], 61, .82)
        pattern(guitar, "MEGALOVANIA Soft Guitar", start, hook[bar], 56, .48)

    answers = [
        [("F4", .5), ("E4", .5), ("D4", .5), (None, 2.5)],
        [("A4", .5), ("G4", .5), ("E4", .5), (None, 2.5)],
        [("G4", .5), ("F4", .5), ("E4", .5), (None, 2.5)],
        [("A4", .5), ("E4", .5), ("D4", .5), (None, 2.5)],
    ]

    for bar in range(BARS):
        start = bar * 4.0
        pattern(context, "ASGORE Pulse 25", start, hook[bar], 86, .82)
        pattern(context, "ASGORE Pulse 12.5", start, hook[bar], 58, .82)
        if bar in (1, 3, 5, 7):
            pattern(context, "MEGALOVANIA Soft Guitar", start + 2.0,
                    answers[(bar - 1) // 2], 46, .40)

        for pos in (0.0, 2.0):
            add(context, "ASGORE Drums", start + pos, 36, .07, 96)
        for pos in (1.0, 3.0):
            add(context, "ASGORE Drums", start + pos, 38, .08, 108)
        for step in range(8):
            add(context, "ASGORE Drums", start + step * .5, 42, .04, 42 if step % 2 else 52)
        if bar in (3, 7):
            for off, drum in ((3.0, 45), (3.25, 47), (3.5, 48), (3.75, 50)):
                add(context, "ASGORE Drums", start + off, drum, .05, 72)

    return pulse, guitar, context


def render_stem(soundfont: Path, preset_data: Preset, events: Sequence[Event]) -> np.ndarray:
    total = int(((BARS * 4 * BEAT) + TAIL) * SR)
    bank, preset, _ = preset_data
    synth = fluidsynth.Synth(gain=.72, samplerate=SR)
    sfid = synth.sfload(str(soundfont))
    if synth.program_select(0, sfid, bank, preset) != 0:
        synth.delete()
        raise RuntimeError(f"Cannot select {bank}:{preset}")
    synth.set_reverb(roomsize=.08, damping=.78, width=.60, level=.035)
    synth.set_chorus(nr=2, level=.025, speed=.18, depth=.75, type=0)

    timeline = []
    for start, duration, note, velocity in events:
        timeline += [(int(start * BEAT * SR), 1, note, velocity),
                     (int((start + duration) * BEAT * SR), 0, note, 0)]
    timeline.sort(key=lambda item: (item[0], item[1]))

    chunks, cursor = [], 0
    for frame, on, note, velocity in timeline:
        if frame > cursor:
            chunks.append(synth.get_samples(frame - cursor))
            cursor = frame
        synth.noteon(0, note, velocity) if on else synth.noteoff(0, note)
    if cursor < total:
        chunks.append(synth.get_samples(total - cursor))
    synth.delete()
    if not chunks:
        return np.zeros((total, 2), dtype=np.float64)
    return (np.concatenate(chunks).astype(np.float64) / 32768.0).reshape(-1, 2)


def write_wav(path: Path, audio: np.ndarray) -> None:
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes(pcm.tobytes())


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        raw = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
    return raw.astype(np.float64).reshape(-1, 2) / 32768.0


def soften_guitar(audio: np.ndarray) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        source, target = Path(tmp) / "raw.wav", Path(tmp) / "soft.wav"
        write_wav(source, audio)
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(source),
            "-af", "highpass=f=170,lowpass=f=6200,equalizer=f=330:t=q:w=1:g=-2.5,equalizer=f=2800:t=q:w=1.1:g=-3.5,acompressor=threshold=-22dB:ratio=1.45:attack=18:release=150",
            "-c:a", "pcm_s16le", str(target),
        ], check=True)
        return read_wav(target)


def mix(soundfont: Path, presets: Dict[str, Preset], events: Dict[str, List[Event]], proof: str) -> np.ndarray:
    total = int(((BARS * 4 * BEAT) + TAIL) * SR)
    output = np.zeros((total, 2), dtype=np.float64)
    gains = {
        "ASGORE Pulse 25": .72 if proof == "context" else .78,
        "ASGORE Pulse 12.5": .46 if proof == "context" else .52,
        "MEGALOVANIA Soft Guitar": .25 if proof == "context" else .56,
        "ASGORE Drums": .59,
    }
    for name, track_events in events.items():
        if not track_events:
            continue
        stem = render_stem(soundfont, presets[name], track_events)
        if name == "MEGALOVANIA Soft Guitar":
            stem = soften_guitar(stem)
        output[:len(stem)] += stem * gains[name]
    peak = float(np.max(np.abs(output)))
    return output / peak * .92 if peak else output


def encode(wav_path: Path, mp3_path: Path) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path),
        "-af", "highpass=f=45,lowpass=f=12500,acompressor=threshold=-18dB:ratio=1.6:attack=12:release=140,alimiter=limit=.96",
        "-codec:a", "libmp3lame", "-q:a", "2", str(mp3_path),
    ], check=True)


def export_midi(path: Path, presets: Dict[str, Preset], events: Dict[str, List[Event]]) -> None:
    midi = MidiFile(ticks_per_beat=TPB)
    tempo = MidiTrack()
    tempo.append(MetaMessage("set_tempo", tempo=bpm2tempo(BPM), time=0))
    midi.tracks.append(tempo)
    channel = 0
    for name, track_events in events.items():
        if not track_events:
            continue
        while channel == 9:
            channel += 1
        ch, channel = channel % 16, channel + 1
        bank, preset, _ = presets[name]
        track = MidiTrack()
        track.append(MetaMessage("track_name", name=name, time=0))
        track.append(Message("control_change", channel=ch, control=0, value=bank, time=0))
        track.append(Message("program_change", channel=ch, program=preset, time=0))
        timeline = []
        for start, duration, note, velocity in track_events:
            timeline += [(round(start * TPB), 1, note, velocity),
                         (round((start + duration) * TPB), 0, note, 0)]
        timeline.sort(key=lambda item: (item[0], item[1]))
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

    presets = resolve_presets(args.soundfont)
    pulse, guitar, context = make_events(tuple(presets))
    for base, events, proof in [
        ("KRIS_ASGORE_PIXEL_LEAD_4BAR", pulse, "pulse"),
        ("KRIS_SOFT_MEGALOVANIA_GUITAR_4BAR", guitar, "guitar"),
        ("KRIS_ASGORE_PIXEL_SOFT_MEGALO_CONTEXT", context, "context"),
    ]:
        wav_path = args.out / f"{base}.wav"
        write_wav(wav_path, mix(args.soundfont, presets, events, proof))
        encode(wav_path, args.out / f"{base}.mp3")

    export_midi(args.out / "KRIS_ASGORE_PIXEL_SOFT_MEGALO.mid", presets, context)
    lines = [
        "144 BPM, D minor, entirely new melody.",
        "Headline ASGORE sound: layered 25% Pulse and 12.5% Pulse.",
        "MEGALOVANIA Overdriven Guitar: shorter, quieter, high-passed and de-harshened.",
        "No ASGORE Piano, Brass, wind lead, strings, violin, or bass layer.",
        "Approved ASGORE drum groove retained.", "", "Resolved presets:",
    ]
    for key, (bank, preset, real_name) in presets.items():
        lines.append(f"- {key}: bank {bank}, preset {preset}, {real_name}")
    (args.out / "KRIS_ASGORE_PIXEL_SOFT_MEGALO_INSTRUMENTS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
