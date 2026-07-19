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
BEAT_SECONDS = 60.0 / BPM
BARS = 8
TAIL_SECONDS = 2.0
TPB = 480

PITCH = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

Event = Tuple[float, float, int, int]
Pattern = Sequence[Tuple[str | None, float]]

# Only the two requested main lead timbres are used for the melody.
PRESETS = {
    "MEGALOVANIA Main Lead": (0, 0, "100 MEGALOVANIA - Overdriven Guitar"),
    "Hopes Main Lead": (3, 8, "087 Hopes and Dreams - Lead Guitar"),
    "MEGALOVANIA Bass": (0, 14, "100 MEGALOVANIA - Bass"),
    "Hopes POWER Drums": (2, 127, "087 Hopes and Dreams - POWER DrumKit"),
}


def note_number(note: str) -> int:
    if len(note) >= 3 and note[1] in "#b":
        name, octave = note[:2], int(note[2:])
    else:
        name, octave = note[0], int(note[1:])
    return 12 * (octave + 1) + PITCH[name]


def add_note(events: Dict[str, List[Event]], track: str, start: float,
             note: str | int, duration: float, velocity: int) -> None:
    pitch = note if isinstance(note, int) else note_number(note)
    events[track].append((start, duration, pitch, velocity))


def add_pattern(events: Dict[str, List[Event]], track: str, start: float,
                pattern: Pattern, velocity: int, transpose: int = 0) -> None:
    cursor = start
    for note, duration in pattern:
        if note is not None:
            add_note(events, track, cursor, note_number(note) + transpose,
                     duration * 0.94, velocity)
        cursor += duration


def build_events() -> tuple[Dict[str, List[Event]], Dict[str, List[Event]]]:
    melody_only = {name: [] for name in PRESETS}
    context = {name: [] for name in PRESETS}

    # Entirely new G-minor hook: four-bar statement, then a higher four-bar answer.
    # No note randomizer, no copied Undertale melody.
    melody: List[Pattern] = [
        [("G4", 1.0), ("Bb4", .5), ("D5", .5), ("C5", 1.0), ("Bb4", 1.0)],
        [("F4", .5), ("G4", .5), ("Bb4", 1.0), ("A4", .5), ("G4", .5), ("D4", 1.0)],
        [("Bb4", .5), ("D5", .5), ("F5", 1.0), ("Eb5", .5), ("D5", .5), ("C5", 1.0)],
        [("A4", .5), ("Bb4", .5), ("C5", 1.0), ("F#4", .5), ("A4", .5), ("G4", 1.0)],
        [("D5", 1.0), ("F5", .5), ("G5", .5), ("F5", 1.0), ("D5", 1.0)],
        [("C5", .5), ("D5", .5), ("F5", 1.0), ("Eb5", .5), ("D5", .5), ("Bb4", 1.0)],
        [("G4", .5), ("Bb4", .5), ("D5", 1.0), ("F5", .5), ("Eb5", .5), ("D5", 1.0)],
        [("C5", .5), ("Bb4", .5), ("A4", .5), ("G4", .5), ("F#4", 1.0), ("G4", 1.0)],
    ]

    roots = ["G2", "Eb2", "Bb1", "D2", "G2", "Eb2", "C2", "D2"]

    for bar, phrase in enumerate(melody):
        start = bar * 4.0
        # Bars 1-4: MEGALOVANIA lead states the idea.
        # Bars 5-8: Hopes and Dreams lead answers and lifts it.
        lead = "MEGALOVANIA Main Lead" if bar < 4 else "Hopes Main Lead"
        add_pattern(melody_only, lead, start, phrase, 112)
        add_pattern(context, lead, start, phrase, 112)

        # Final two bars briefly combine both requested leads, with no third lead sound.
        if bar >= 6:
            other = "Hopes Main Lead" if lead == "MEGALOVANIA Main Lead" else "MEGALOVANIA Main Lead"
            add_pattern(context, other, start, phrase, 54, transpose=-12)

        root = note_number(roots[bar])
        bass_pattern = [root, root + 7, root + 12, root + 7,
                        root, root + 7, root + 12, root + 7]
        for step, pitch in enumerate(bass_pattern):
            add_note(context, "MEGALOVANIA Bass", start + step * .5,
                     pitch, .38, 92 if step in (0, 4) else 72)

        # Straight POWER rock beat. No fast fill spam.
        for pos in (0.0, 2.0):
            add_note(context, "Hopes POWER Drums", start + pos, 36, .08, 110)
        for pos in (1.0, 3.0):
            add_note(context, "Hopes POWER Drums", start + pos, 38, .08, 116)
        for step in range(8):
            add_note(context, "Hopes POWER Drums", start + step * .5, 42, .04,
                     55 if step % 2 else 64)
        if bar in (3, 7):
            for off, drum in ((3.0, 45), (3.25, 47), (3.5, 48), (3.75, 50)):
                add_note(context, "Hopes POWER Drums", start + off, drum, .05, 78)

    return melody_only, context


def render_stem(soundfont: Path, bank: int, preset: int,
                events: Sequence[Event], gain: float = .82) -> np.ndarray:
    total_frames = int(((BARS * 4 * BEAT_SECONDS) + TAIL_SECONDS) * SR)
    synth = fluidsynth.Synth(gain=gain, samplerate=SR)
    sfid = synth.sfload(str(soundfont))
    result = synth.program_select(0, sfid, bank, preset)
    if result != 0:
        synth.delete()
        raise RuntimeError(f"Cannot select bank={bank}, preset={preset}")
    synth.set_reverb(roomsize=.16, damping=.72, width=.72, level=.08)
    synth.set_chorus(nr=2, level=.10, speed=.25, depth=1.5, type=0)

    timeline = []
    for start, duration, note, velocity in events:
        timeline.append((int(start * BEAT_SECONDS * SR), 1, note, velocity))
        timeline.append((int((start + duration) * BEAT_SECONDS * SR), 0, note, 0))
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


def mix(soundfont: Path, events: Dict[str, List[Event]]) -> np.ndarray:
    total_frames = int(((BARS * 4 * BEAT_SECONDS) + TAIL_SECONDS) * SR)
    result = np.zeros((total_frames, 2), dtype=np.float64)
    gains = {
        "MEGALOVANIA Main Lead": .82,
        "Hopes Main Lead": .82,
        "MEGALOVANIA Bass": .64,
        "Hopes POWER Drums": .70,
    }
    for name, track_events in events.items():
        if not track_events:
            continue
        bank, preset, _ = PRESETS[name]
        stem = render_stem(soundfont, bank, preset, track_events)
        result[:len(stem)] += stem * gains[name]
    peak = float(np.max(np.abs(result)))
    if peak > 0:
        result = result / peak * .94
    return result


def write_wav(path: Path, audio: np.ndarray) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes(pcm.tobytes())


def encode_mp3(wav_path: Path, mp3_path: Path) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path),
        "-af", "highpass=f=35,lowpass=f=13000,acompressor=threshold=-18dB:ratio=2.2:attack=8:release=120,alimiter=limit=0.96",
        "-codec:a", "libmp3lame", "-q:a", "2", str(mp3_path),
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
        while channel == 9:
            channel += 1
        midi_channel = channel % 16
        channel += 1
        track = MidiTrack()
        track.append(MetaMessage("track_name", name=name, time=0))
        bank, preset, _ = PRESETS[name]
        track.append(Message("control_change", channel=midi_channel,
                             control=0, value=min(127, bank), time=0))
        track.append(Message("program_change", channel=midi_channel,
                             program=min(127, preset), time=0))
        timeline = []
        for start, duration, note, velocity in track_events:
            timeline.append((round(start * TPB), 1, note, velocity))
            timeline.append((round((start + duration) * TPB), 0, note, 0))
        timeline.sort(key=lambda item: (item[0], item[1]))
        previous = 0
        for tick, is_on, note, velocity in timeline:
            track.append(Message("note_on" if is_on else "note_off",
                                 channel=midi_channel, note=note,
                                 velocity=velocity if is_on else 0,
                                 time=tick - previous))
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

    melody_events, context_events = build_events()
    melody_audio = mix(args.soundfont, melody_events)
    context_audio = mix(args.soundfont, context_events)

    melody_wav = args.out / "KRIS_TWO_MAIN_LEADS_MELODY.wav"
    context_wav = args.out / "KRIS_TWO_MAIN_LEADS_CONTEXT.wav"
    write_wav(melody_wav, melody_audio)
    write_wav(context_wav, context_audio)
    encode_mp3(melody_wav, args.out / "KRIS_TWO_MAIN_LEADS_MELODY.mp3")
    encode_mp3(context_wav, args.out / "KRIS_TWO_MAIN_LEADS_CONTEXT.mp3")
    export_midi(args.out / "KRIS_TWO_MAIN_LEADS.mid", context_events)

    lines = [
        f"BPM: {BPM}",
        "Key: G minor",
        "Melody lead 1: 100 MEGALOVANIA - Overdriven Guitar (bank 0, preset 0)",
        "Melody lead 2: 087 Hopes and Dreams - Lead Guitar (bank 3, preset 8)",
        "Support only: 100 MEGALOVANIA - Bass; 087 Hopes and Dreams - POWER DrumKit",
        "No piano lead, square lead, pulse lead, wind, brass, violin, strings, choir, or organ.",
        "The melody is original and manually written; it does not copy Undertale notes.",
    ]
    (args.out / "KRIS_TWO_MAIN_LEADS_INSTRUMENTS.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
