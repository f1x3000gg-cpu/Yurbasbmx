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
BPM = 144
BEAT_SECONDS = 60.0 / BPM
BARS = 8
TAIL_SECONDS = 2.2
TPB = 480

PITCH = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

Event = Tuple[float, float, int, int]
Pattern = Sequence[Tuple[str | int | None, float]]

# Only the requested recognizable battle colours are active.
PRESETS = {
    "MEGALOVANIA Low Guitar": (0, 0, "100 MEGALOVANIA - Overdriven Guitar"),
    "MEGALOVANIA Bass": (0, 14, "100 MEGALOVANIA - Bass"),
    "ASGORE Main Brass": (2, 78, "077 ASGORE - Brass"),
    "ASGORE Orchestra Hit": (2, 77, "077 ASGORE - Orchestra Hit"),
    "ASGORE Timpani": (2, 74, "077 ASGORE - Timpani"),
    "ASGORE Drums": (2, 80, "077 ASGORE - Drumkit"),
}


def note_number(note: str | int) -> int:
    if isinstance(note, int):
        return note
    if len(note) >= 3 and note[1] in "#b":
        name, octave = note[:2], int(note[2:])
    else:
        name, octave = note[0], int(note[1:])
    return 12 * (octave + 1) + PITCH[name]


def add_note(events: Dict[str, List[Event]], track: str, start: float,
             note: str | int, duration: float, velocity: int) -> None:
    pitch = note if isinstance(note, int) else note_number(note)
    events[track].append((float(start), float(duration), int(pitch), int(velocity)))


def add_pattern(events: Dict[str, List[Event]], track: str, start: float,
                pattern: Pattern, velocity: int, transpose: int = 0) -> None:
    cursor = start
    for note, duration in pattern:
        if note is not None:
            add_note(events, track, cursor, note_number(note) + transpose,
                     max(.05, duration * .91), velocity)
        cursor += duration


def empty_events() -> Dict[str, List[Event]]:
    return {name: [] for name in PRESETS}


def build_events() -> tuple[Dict[str, List[Event]], Dict[str, List[Event]], Dict[str, List[Event]]]:
    megalo_only = empty_events()
    asgore_only = empty_events()
    context = empty_events()

    # One low A-minor battle motif. Bars 1-4 use the MEGALOVANIA guitar in a
    # deliberately lower register. Bars 5-8 repeat and complete the SAME motif
    # with the clearly audible ASGORE Brass preset; it is not a second song.
    phrase: List[Pattern] = [
        [("A3", .5), ("A3", .5), ("C4", .5), ("E4", .5), ("D4", 1.0), ("C4", 1.0)],
        [("G3", .5), ("A3", .5), ("C4", 1.0), ("B3", .5), ("A3", .5), ("E3", 1.0)],
        [("C4", .5), ("E4", .5), ("G4", 1.0), ("F4", .5), ("E4", .5), ("D4", 1.0)],
        [("B3", .5), ("C4", .5), ("D4", 1.0), ("G#3", .5), ("B3", .5), ("A3", 1.0)],
    ]

    roots = ["A1", "F1", "C2", "E1", "A1", "F1", "D2", "E1"]

    for bar in range(BARS):
        start = bar * 4.0
        pattern = phrase[bar % 4]

        if bar < 4:
            add_pattern(megalo_only, "MEGALOVANIA Low Guitar", start, pattern, 108)
            add_pattern(context, "MEGALOVANIA Low Guitar", start, pattern, 108)
        else:
            # This is the actual ASGORE brass colour, placed in front of the mix.
            add_pattern(asgore_only, "ASGORE Main Brass", start, pattern, 116)
            add_pattern(context, "ASGORE Main Brass", start, pattern, 116)
            # A quiet low guitar shadow keeps the entire eight bars related.
            add_pattern(context, "MEGALOVANIA Low Guitar", start, pattern, 42, transpose=-12)

        root = note_number(roots[bar])
        bass_line = [root, root + 7, root + 12, root + 7,
                     root, root + 7, root + 12, root + 7]
        for step, pitch in enumerate(bass_line):
            add_note(context, "MEGALOVANIA Bass", start + step * .5,
                     pitch, .34, 92 if step in (0, 4) else 70)

        # Direct ASGORE drum groove. The kit is deliberately louder than before.
        for pos in (0.0, 2.0):
            add_note(context, "ASGORE Drums", start + pos, 36, .08, 114)
        for pos in (1.0, 3.0):
            add_note(context, "ASGORE Drums", start + pos, 38, .09, 120)
        for step in range(8):
            add_note(context, "ASGORE Drums", start + step * .5, 42, .04,
                     50 if step % 2 else 61)
        if bar in (3, 7):
            for off, drum in ((3.0, 45), (3.25, 47), (3.5, 48), (3.75, 50)):
                add_note(context, "ASGORE Drums", start + off, drum, .05, 84)

        # Recognizable ASGORE weight at phrase boundaries.
        if bar in (0, 4):
            add_note(context, "ASGORE Orchestra Hit", start, root + 24, .22, 108)
            add_note(context, "ASGORE Timpani", start, root + 12, .55, 106)
        elif bar in (3, 7):
            add_note(context, "ASGORE Timpani", start + 3.0, root + 12, .46, 92)

    return megalo_only, asgore_only, context


def render_stem(soundfont: Path, bank: int, preset: int,
                events: Sequence[Event], gain: float = .82) -> np.ndarray:
    total_frames = int(((BARS * 4 * BEAT_SECONDS) + TAIL_SECONDS) * SR)
    synth = fluidsynth.Synth(gain=gain, samplerate=SR)
    sfid = synth.sfload(str(soundfont))
    result = synth.program_select(0, sfid, bank, preset)
    if result != 0:
        synth.delete()
        raise RuntimeError(f"Cannot select bank={bank}, preset={preset}")
    synth.set_reverb(roomsize=.12, damping=.72, width=.66, level=.055)
    synth.set_chorus(nr=2, level=.06, speed=.22, depth=1.2, type=0)

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


def mix(soundfont: Path, events: Dict[str, List[Event]], gains: Dict[str, float]) -> np.ndarray:
    total_frames = int(((BARS * 4 * BEAT_SECONDS) + TAIL_SECONDS) * SR)
    result = np.zeros((total_frames, 2), dtype=np.float64)
    for name, track_events in events.items():
        if not track_events or gains.get(name, 0.0) <= 0:
            continue
        bank, preset, _ = PRESETS[name]
        stem = render_stem(soundfont, bank, preset, track_events)
        result[:len(stem)] += stem * gains[name]
    result = np.tanh(result * 1.02)
    peak = float(np.max(np.abs(result)))
    if peak > 0:
        result = result / peak * .93
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
        "-af", "highpass=f=28,lowpass=f=11800,acompressor=threshold=-18dB:ratio=2:attack=9:release=125,alimiter=limit=0.96",
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
        bank, preset, _ = PRESETS[name]
        track = MidiTrack()
        track.append(MetaMessage("track_name", name=name, time=0))
        track.append(Message("control_change", channel=midi_channel, control=0,
                             value=min(127, bank), time=0))
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
    parser.add_argument("--soundfont", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    megalo_events, asgore_events, context_events = build_events()

    megalo_audio = mix(args.soundfont, megalo_events, {
        "MEGALOVANIA Low Guitar": .92,
    })
    asgore_audio = mix(args.soundfont, asgore_events, {
        "ASGORE Main Brass": .96,
    })
    context_audio = mix(args.soundfont, context_events, {
        "MEGALOVANIA Low Guitar": .76,
        "MEGALOVANIA Bass": .56,
        "ASGORE Main Brass": .90,
        "ASGORE Orchestra Hit": .55,
        "ASGORE Timpani": .62,
        "ASGORE Drums": .72,
    })

    megalo_wav = args.out / "KRIS_LOW_MEGALOVANIA_ONLY.wav"
    asgore_wav = args.out / "KRIS_ASGORE_MAIN_SOUND_ONLY.wav"
    context_wav = args.out / "KRIS_LOW_MEGALO_CLEAR_ASGORE_CONTEXT.wav"
    write_wav(megalo_wav, megalo_audio)
    write_wav(asgore_wav, asgore_audio)
    write_wav(context_wav, context_audio)
    encode_mp3(megalo_wav, args.out / "KRIS_LOW_MEGALOVANIA_ONLY.mp3")
    encode_mp3(asgore_wav, args.out / "KRIS_ASGORE_MAIN_SOUND_ONLY.mp3")
    encode_mp3(context_wav, args.out / "KRIS_LOW_MEGALO_CLEAR_ASGORE_CONTEXT.mp3")
    export_midi(args.out / "KRIS_LOW_MEGALO_CLEAR_ASGORE.mid", context_events)

    lines = [
        f"BPM: {BPM}",
        "Key: A minor",
        "MEGALOVANIA lead is one octave/lower-register compared with the rejected proof.",
        "Prominent ASGORE lead: 077 ASGORE - Brass (bank 2, preset 78).",
        "Additional clearly audible ASGORE colours: Orchestra Hit, Timpani, Drumkit.",
        "No Hopes and Dreams Lead Guitar. No saxophone, clarinet, flute, violin, or choir.",
        "The same four-bar motif is passed from MEGALOVANIA guitar to ASGORE brass; there is no unrelated second melody.",
    ]
    (args.out / "KRIS_LOW_MEGALO_CLEAR_ASGORE_INSTRUMENTS.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
