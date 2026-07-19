#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import fluidsynth
import numpy as np
from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo

SR = 44100
BPM = 164
BEAT = 60.0 / BPM
BARS = 40
TAIL = 2.2
TPB = 480

PITCH = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

Event = Tuple[float, float, int, int]
Pattern = Sequence[Tuple[str | None, float]]
Preset = Tuple[int, int, str]

PRESETS: Dict[str, Preset] = {
    "MEGALOVANIA Lead Guitar": (0, 0, "100 MEGALOVANIA - Over. Gt."),
    "MEGALOVANIA Background Guitar": (0, 13, "100 MEGALOVANIA - Bg. Gt."),
    "MEGALOVANIA Bass": (0, 14, "100 MEGALOVANIA - Bass"),
    "ASGORE Timpani": (2, 74, "077 ASGORE - Timpani"),
    "ASGORE Choir": (2, 76, "077 ASGORE - Choir Aahs"),
    "ASGORE Orchestra Hit": (2, 77, "077 ASGORE - Orchestra Hit"),
    "ASGORE Drums": (2, 80, "077 ASGORE - Drumkit"),
}


def nn(note: str) -> int:
    if len(note) >= 3 and note[1] in "#b":
        name, octave = note[:2], int(note[2:])
    else:
        name, octave = note[0], int(note[1:])
    return 12 * (octave + 1) + PITCH[name]


def add(events: Dict[str, List[Event]], track: str, start: float,
        note: str | int, duration: float, velocity: int) -> None:
    pitch = note if isinstance(note, int) else nn(note)
    events[track].append((float(start), float(duration), int(pitch), int(velocity)))


def pattern(events: Dict[str, List[Event]], track: str, start: float,
            notes: Pattern, velocity: int, gate: float = .72) -> None:
    cursor = start
    for note, duration in notes:
        if note is not None:
            add(events, track, cursor, note, max(.04, duration * gate), velocity)
        cursor += duration
    if abs(cursor - 4.0) > .001:
        raise ValueError(f"Pattern length must be 4 beats, got {cursor}")


def chord(events: Dict[str, List[Event]], track: str, start: float,
          notes: Sequence[str], duration: float, velocity: int) -> None:
    for note in notes:
        add(events, track, start, note, duration, velocity)


def melody_bars() -> List[Pattern]:
    return [
        [("A4", .5), ("C5", .5), ("D5", .75), ("F5", .25), ("E5", .5), ("D5", .5), ("A4", 1.0)],
        [("G4", .5), ("A4", .25), ("C5", .25), ("E5", .75), ("D5", .25), ("C5", .5), ("A4", .5), ("G4", 1.0)],
        [("F4", .5), ("A4", .5), ("C5", .5), ("D5", .5), ("C5", .5), ("A4", .5), ("G4", .5), ("F4", .5)],
        [("E4", .5), ("G4", .5), ("A4", .75), ("C#5", .25), ("D5", .5), ("C#5", .5), ("A4", 1.0)],
        [("D5", .25), ("F5", .25), ("A5", .5), ("G5", .5), ("F5", .5), ("E5", .5), ("D5", .5), ("C5", 1.0)],
        [("Bb4", .5), ("D5", .5), ("G5", .5), ("F5", .5), ("D5", .5), ("C5", .5), ("Bb4", .5), ("A4", .5)],
        [("F4", .25), ("A4", .25), ("C5", .5), ("F5", .75), ("E5", .25), ("D5", .5), ("C5", .5), ("A4", 1.0)],
        [("E5", .5), ("C#5", .5), ("A4", .5), ("C#5", .5), ("D5", .5), ("F5", .5), ("E5", .5), ("D5", .5)],
    ]


def variation_bars() -> List[Pattern]:
    return [
        [("D5", .25), ("E5", .25), ("F5", .5), ("A5", .5), ("G5", .5), ("F5", .25), ("E5", .25), ("D5", .5), ("A4", 1.0)],
        [("C5", .5), ("E5", .25), ("G5", .25), ("A5", .5), ("G5", .5), ("E5", .5), ("D5", .5), ("C5", 1.0)],
        [("A4", .25), ("C5", .25), ("F5", .5), ("E5", .5), ("C5", .5), ("D5", .5), ("A4", .5), ("C5", 1.0)],
        [("C#5", .25), ("E5", .25), ("A5", .5), ("G5", .5), ("E5", .5), ("C#5", .5), ("A4", .5), ("E5", 1.0)],
        [("F5", .25), ("A5", .25), ("C6", .5), ("A5", .5), ("G5", .5), ("F5", .5), ("E5", .5), ("D5", 1.0)],
        [("D5", .25), ("G5", .25), ("Bb5", .5), ("A5", .5), ("G5", .5), ("F5", .5), ("D5", .5), ("Bb4", 1.0)],
        [("C5", .25), ("F5", .25), ("A5", .5), ("G5", .5), ("F5", .5), ("E5", .5), ("C5", .5), ("A4", 1.0)],
        [("A4", .25), ("C#5", .25), ("E5", .5), ("A5", .5), ("G5", .5), ("E5", .5), ("C#5", .5), ("D5", 1.0)],
    ]


def breakdown_bars() -> List[Pattern]:
    return [
        [("D5", 1.0), (None, .5), ("A4", .5), ("C5", .5), ("D5", .5), ("F5", 1.0)],
        [("C5", .5), (None, .5), ("G4", 1.0), ("C5", .5), ("E5", .5), ("D5", 1.0)],
        [("Bb4", 1.0), ("D5", .5), (None, .5), ("F5", .5), ("E5", .5), ("D5", 1.0)],
        [("A4", .5), ("C#5", .5), ("E5", 1.0), ("G5", .5), ("E5", .5), ("D5", 1.0)],
    ]


def add_drums(events: Dict[str, List[Event]], bar: int, intensity: int) -> None:
    start = bar * 4.0
    drum = "ASGORE Drums"
    if bar in (0, 8, 16, 24, 28, 32):
        add(events, drum, start, 49, .12, 112)

    if intensity == 0:
        for pos in (0.0, 1.5, 2.75):
            add(events, drum, start + pos, 36, .07, 92)
        add(events, drum, start + 2.0, 38, .09, 112)
        for step in range(8):
            add(events, drum, start + step * .5, 42, .04, 48 if step % 2 else 60)
    else:
        kick_positions = [0.0, .75, 2.0, 2.75] if intensity == 1 else [0.0, .5, 1.5, 2.0, 2.5, 3.25]
        for pos in kick_positions:
            add(events, drum, start + pos, 36, .06, 98 if intensity == 1 else 108)
        for pos in (1.0, 3.0):
            add(events, drum, start + pos, 38, .08, 112)
        for step in range(16):
            note = 46 if intensity == 2 and step in (7, 15) else 42
            add(events, drum, start + step * .25, note, .03, 42 + (step % 4 == 0) * 18)
        if intensity == 2:
            for pos in (1.75, 3.75):
                add(events, drum, start + pos, 38, .045, 72)

    if bar % 8 == 7 or bar in (27, 31, 39):
        for off, note, vel in ((3.0, 45, 78), (3.25, 47, 86), (3.5, 48, 94), (3.75, 50, 108)):
            add(events, drum, start + off, note, .05, vel)


def build_events() -> Dict[str, List[Event]]:
    events = {name: [] for name in PRESETS}
    hook = melody_bars()
    variation = variation_bars()
    breakdown = breakdown_bars()
    roots = ["D2", "C2", "Bb1", "A1", "D2", "G1", "F2", "A1"]
    choir_chords = [
        ["D3", "A3", "D4", "F4"], ["C3", "G3", "C4", "E4"],
        ["Bb2", "F3", "Bb3", "D4"], ["A2", "E3", "A3", "C#4"],
        ["D3", "A3", "D4", "F4"], ["G2", "D3", "G3", "Bb3"],
        ["F3", "C4", "F4", "A4"], ["A2", "E3", "A3", "C#4"],
    ]

    for bar in range(BARS):
        start = bar * 4.0
        slot = bar % 8

        if bar < 8:
            lead_pattern, velocity, gate = hook[slot], 82, .66
            intensity = 1
        elif bar < 16:
            lead_pattern, velocity, gate = variation[slot], 86, .64
            intensity = 1
        elif bar < 24:
            lead_pattern, velocity, gate = variation[(slot + 3) % 8], 94, .61
            intensity = 2
        elif bar < 28:
            lead_pattern, velocity, gate = breakdown[bar - 24], 78, .74
            intensity = 0
        elif bar < 32:
            lead_pattern, velocity, gate = hook[(slot + 4) % 8], 90, .62
            intensity = 2
        else:
            lead_pattern, velocity, gate = variation[(slot + 1) % 8], 98, .59
            intensity = 2

        pattern(events, "MEGALOVANIA Lead Guitar", start, lead_pattern, velocity, gate)

        bg_notes = ["D3", "C3", "Bb2", "A2", "D3", "G2", "F3", "A2"]
        bg = bg_notes[slot]
        for step in range(8):
            if 24 <= bar < 28 and step % 2:
                continue
            add(events, "MEGALOVANIA Background Guitar", start + step * .5,
                bg, .22 if bar < 16 else .28, 42 if bar < 16 else 50)

        root = nn(roots[slot])
        bass_positions = (0.0, 1.5, 2.0, 3.25) if intensity != 2 else (0.0, .75, 1.5, 2.0, 2.75, 3.5)
        for i, pos in enumerate(bass_positions):
            add(events, "MEGALOVANIA Bass", start + pos,
                root + (7 if i % 3 == 2 else 0), .32, 54 if intensity < 2 else 62)

        if bar >= 12:
            choir_vel = 34 if bar < 32 else 46
            chord(events, "ASGORE Choir", start, choir_chords[slot], 3.55, choir_vel)
        if bar in (0, 8, 16, 20, 24, 28, 32, 36, 39):
            hit_note = roots[slot].replace("2", "4").replace("1", "3")
            add(events, "ASGORE Orchestra Hit", start, hit_note, .38,
                72 if bar < 32 else 92)
        if intensity == 2 or bar in (0, 8, 24, 28):
            add(events, "ASGORE Timpani", start, root + 12, .42,
                72 if intensity < 2 else 90)
            if intensity == 2:
                add(events, "ASGORE Timpani", start + 2.75, root + 19, .32, 70)

        add_drums(events, bar, intensity)

    return events


def render_stem(soundfont: Path, preset_data: Preset,
                events: Sequence[Event]) -> np.ndarray:
    total = int(((BARS * 4 * BEAT) + TAIL) * SR)
    bank, preset, _ = preset_data
    synth = fluidsynth.Synth(gain=.72, samplerate=SR)
    sfid = synth.sfload(str(soundfont))
    if synth.program_select(0, sfid, bank, preset) != 0:
        synth.delete()
        raise RuntimeError(f"Cannot select bank={bank} preset={preset}")
    synth.set_reverb(roomsize=.10, damping=.75, width=.68, level=.04)
    synth.set_chorus(nr=2, level=.025, speed=.22, depth=.8, type=0)

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


def write_wav(path: Path, audio: np.ndarray) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes(pcm.tobytes())


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        raw = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
        channels = handle.getnchannels()
    return raw.astype(np.float64).reshape(-1, channels) / 32768.0


def process_guitar(audio: np.ndarray, background: bool) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "raw.wav"
        target = Path(tmp) / "processed.wav"
        write_wav(source, audio)
        if background:
            filt = (
                "highpass=f=150,lowpass=f=6100,"
                "equalizer=f=300:t=q:w=1.0:g=-1.5,"
                "equalizer=f=2700:t=q:w=1.2:g=-2.5,"
                "acompressor=threshold=-22dB:ratio=1.5:attack=15:release=120"
            )
        else:
            filt = (
                "highpass=f=105,lowpass=f=7900,"
                "equalizer=f=260:t=q:w=1.0:g=-1.2,"
                "equalizer=f=2850:t=q:w=1.1:g=-2.0,"
                "equalizer=f=4700:t=q:w=1.0:g=-1.2,"
                "acompressor=threshold=-20dB:ratio=1.35:attack=18:release=140"
            )
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(source),
            "-af", filt, "-c:a", "pcm_s16le", str(target),
        ], check=True)
        return read_wav(target)


def mix(soundfont: Path, events: Dict[str, List[Event]]) -> np.ndarray:
    total = int(((BARS * 4 * BEAT) + TAIL) * SR)
    output = np.zeros((total, 2), dtype=np.float64)
    gains = {
        "MEGALOVANIA Lead Guitar": .60,
        "MEGALOVANIA Background Guitar": .25,
        "MEGALOVANIA Bass": .27,
        "ASGORE Timpani": .34,
        "ASGORE Choir": .20,
        "ASGORE Orchestra Hit": .31,
        "ASGORE Drums": .66,
    }

    for name, track_events in events.items():
        if not track_events:
            continue
        stem = render_stem(soundfont, PRESETS[name], track_events)
        if name == "MEGALOVANIA Lead Guitar":
            stem = process_guitar(stem, background=False)
        elif name == "MEGALOVANIA Background Guitar":
            stem = process_guitar(stem, background=True)
        output[:len(stem)] += stem * gains[name]

    peak = float(np.max(np.abs(output)))
    if peak > 0:
        output = output / peak * .91
    return output


def encode(wav_path: Path, mp3_path: Path, ogg_path: Path) -> None:
    master = (
        "highpass=f=42,lowpass=f=12500,"
        "acompressor=threshold=-17dB:ratio=1.7:attack=9:release=115,"
        "alimiter=limit=.95"
    )
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path),
        "-af", master, "-codec:a", "libmp3lame", "-q:a", "2", str(mp3_path),
    ], check=True)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path),
        "-af", master, "-codec:a", "libvorbis", "-q:a", "6", str(ogg_path),
    ], check=True)


def export_midi(path: Path, events: Dict[str, List[Event]]) -> None:
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
        bank, preset, _ = PRESETS[name]
        track = MidiTrack()
        track.append(MetaMessage("track_name", name=name, time=0))
        track.append(Message("control_change", channel=ch, control=0, value=bank, time=0))
        track.append(Message("program_change", channel=ch, program=preset, time=0))
        timeline = []
        for start, duration, note, velocity in track_events:
            timeline.append((round(start * TPB), 1, note, velocity))
            timeline.append((round((start + duration) * TPB), 0, note, 0))
        timeline.sort(key=lambda item: (item[0], item[1]))
        previous = 0
        for tick, on, note, velocity in timeline:
            track.append(Message(
                "note_on" if on else "note_off", channel=ch,
                note=note, velocity=velocity if on else 0,
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

    events = build_events()
    audio = mix(args.soundfont, events)
    wav_path = args.out / "KRIS_AGGRESSIVE_MINUTE_FULL.wav"
    mp3_path = args.out / "KRIS_AGGRESSIVE_MINUTE_FULL.mp3"
    ogg_path = args.out / "KRIS_AGGRESSIVE_MINUTE_GAME.ogg"
    midi_path = args.out / "KRIS_AGGRESSIVE_MINUTE.mid"
    info_path = args.out / "KRIS_AGGRESSIVE_MINUTE_INSTRUMENTS.txt"

    write_wav(wav_path, audio)
    encode(wav_path, mp3_path, ogg_path)
    export_midi(midi_path, events)
    duration = BARS * 4 * BEAT + TAIL
    info_path.write_text(
        "KRIS AGGRESSIVE MINUTE\n"
        f"Tempo: {BPM} BPM\nBars: {BARS}\nApprox duration: {duration:.1f} seconds\n\n"
        "Main lead:\n"
        "- 100 MEGALOVANIA - Over. Gt. (original compiled track preset, slightly softened by EQ)\n\n"
        "Supporting MEGALOVANIA instruments:\n"
        "- 100 MEGALOVANIA - Bg. Gt.\n"
        "- 100 MEGALOVANIA - Bass\n\n"
        "ASGORE instruments (no piano, no pulse lead, no brass/woodwind lead):\n"
        "- 077 ASGORE - Drumkit\n"
        "- 077 ASGORE - Timpani\n"
        "- 077 ASGORE - Choir Aahs\n"
        "- 077 ASGORE - Orchestra Hit\n\n"
        "Structure:\n"
        "- Bars 1-8: immediate hook\n"
        "- Bars 9-16: note/rhythm variation\n"
        "- Bars 17-24: aggressive chorus\n"
        "- Bars 25-28: half-time break\n"
        "- Bars 29-32: rebuild\n"
        "- Bars 33-40: final climax\n",
        encoding="utf-8",
    )
    print(f"Rendered {duration:.1f}s to {args.out}")


if __name__ == "__main__":
    main()
