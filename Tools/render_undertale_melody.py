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
from sf2utils.sf2parse import Sf2File

SR = 44100
BPM = 166
BEAT = 60.0 / BPM
BAR = 4.0 * BEAT
BARS = 32
TAIL = 2.2
TPB = 480

PITCH = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

Event = Tuple[float, float, int, int]
Pattern = Sequence[Tuple[str | int | None, float]]


def note_number(note: str | int) -> int:
    if isinstance(note, int):
        return note
    if len(note) >= 3 and note[1] in "#b":
        pitch, octave = note[:2], int(note[2:])
    else:
        pitch, octave = note[0], int(note[1:])
    return 12 * (octave + 1) + PITCH[pitch]


def note_hz(note: int) -> float:
    return 440.0 * 2.0 ** ((note - 69) / 12.0)


def add_note(events: Dict[str, List[Event]], track: str, start: float,
             note: str | int, duration: float, velocity: int) -> None:
    events[track].append((start, duration, note_number(note), velocity))


def add_pattern(events: Dict[str, List[Event]], track: str, start: float,
                pattern: Pattern, velocity: int, shift: int = 0) -> None:
    cursor = start
    for note, duration in pattern:
        if note is not None:
            add_note(events, track, cursor, note_number(note) + shift,
                     duration * 0.93, velocity)
        cursor += duration


def add_chord(events: Dict[str, List[Event]], track: str, start: float,
              notes: Sequence[str | int], duration: float, velocity: int) -> None:
    for note in notes:
        add_note(events, track, start, note, duration, velocity)


def preset_name(preset) -> str:
    value = getattr(preset, "name", "")
    if isinstance(value, bytes):
        return value.decode("latin1", errors="replace")
    return str(value)


def preset_bank(preset) -> int:
    return int(getattr(preset, "bank", 0))


def preset_program(preset) -> int:
    return int(getattr(preset, "preset", getattr(preset, "program", 0)))


def find_preset(presets, alternatives: Sequence[Sequence[str]],
                fallback_bank: int, fallback_program: int,
                drums: bool = False):
    rows = []
    for preset in presets:
        name = preset_name(preset)
        normalized = name.casefold().replace("_", " ").replace("-", " ")
        rows.append((normalized, name, preset_bank(preset), preset_program(preset)))

    for words in alternatives:
        lowered = [word.casefold() for word in words]
        matches = []
        for normalized, name, bank, program in rows:
            if drums and bank < 120:
                continue
            if all(word in normalized for word in lowered):
                matches.append((len(normalized), name, bank, program))
        if matches:
            _, name, bank, program = sorted(matches)[0]
            return name, bank, program
    return f"fallback bank={fallback_bank} preset={fallback_program}", fallback_bank, fallback_program


def build_preset_map(soundfont: Path):
    with soundfont.open("rb") as handle:
        sf2 = Sf2File(handle)
        presets = list(sf2.presets)

    return {
        "Clav": find_preset(presets, [["comp", "clav"], ["clav"]], 0, 7),
        "Baritone Sax": find_preset(presets, [["baritone", "sax"]], 0, 67),
        "Romantic Trumpet": find_preset(
            presets, [["romantic", "tp"], ["romantic", "trumpet"], ["trumpet"]], 0, 56
        ),
        "Fretless Bass": find_preset(presets, [["fretless", "bass"], ["fretless"]], 0, 35),
        "Power Guitar": find_preset(presets, [["power", "guitar"], ["distortion", "guitar"]], 16, 30),
        "Strings": find_preset(presets, [["strings"]], 0, 48),
        "Choir Aahs": find_preset(presets, [["choir", "aahs"], ["aah", "choir"], ["choir"]], 0, 52),
        "Orchestra Hit": find_preset(presets, [["orchestra", "hit"], ["orchestrahit"], ["impact", "hit"]], 0, 55),
        "Timpani": find_preset(presets, [["timpani"]], 0, 47),
        "Drums": find_preset(presets, [["power"], ["standard", "1"], ["standard"]], 128, 16, drums=True),
    }


def build_events() -> Dict[str, List[Event]]:
    track_names = [
        "Clav", "Baritone Sax", "Romantic Trumpet", "Fretless Bass",
        "Power Guitar", "Strings", "Choir Aahs", "Orchestra Hit",
        "Timpani", "Drums", "25% Pulse",
    ]
    events: Dict[str, List[Event]] = {name: [] for name in track_names}

    # B minor with one borrowed G-natural/F-natural colour. The theme is a
    # written eight-bar phrase: question (1-2), expansion (3-4), chorus-like
    # answer (5-6), cadence (7-8). It is not generated or transposed at random.
    hook: List[Pattern] = [
        [(None, .5), ("F#4", .5), ("B4", .5), ("C#5", .5), ("D5", 1), ("A4", .5), ("F#4", .5)],
        [("E4", .5), ("F#4", .5), ("A4", 1), ("G4", .5), ("E4", .5), ("F#4", 1)],
        [("B4", .5), ("B4", .5), ("D5", .5), ("E5", .5), ("F5", .5), ("E5", .5), ("D5", 1)],
        [("C#5", .5), ("A4", .5), ("F#4", 1), ("G4", .5), ("A4", .5), ("B4", 1)],
        [("D5", .5), ("F5", .5), ("E5", 1), ("C#5", .5), ("B4", .5), ("A4", 1)],
        [("G4", .5), ("B4", .5), ("D5", 1), ("C#5", .5), ("A4", .5), ("F#4", 1)],
        [("B4", .5), ("A4", .5), ("F#4", .5), ("E4", .5), ("G4", .5), ("A4", .5), ("C#5", 1)],
        [("B4", .5), ("A4", .5), ("F#4", 1), ("C#5", .5), ("D5", .5), ("B4", 1)],
    ]

    # A real contrasting section: broader rhythm, higher tension, then a clear
    # return point. It is not the hook shifted to a new key.
    bridge: List[Pattern] = [
        [("F#4", 1), ("A4", .5), ("B4", .5), ("D5", 1), ("C#5", 1)],
        [("B4", .5), ("A4", .5), ("F#4", 1), ("E4", 1), (None, 1)],
        [("G4", .5), ("A4", .5), ("B4", 1), ("D5", .5), ("C#5", .5), ("A4", 1)],
        [("F#4", 1), ("C#5", .5), ("D5", .5), ("E5", 1), ("D5", 1)],
        [("B4", .5), ("D5", .5), ("F5", 1), ("E5", .5), ("D5", .5), ("B4", 1)],
        [("A4", .5), ("B4", .5), ("C#5", 1), ("D5", .5), ("C#5", .5), ("A4", 1)],
        [("G4", 1), ("B4", .5), ("A4", .5), ("F#4", 1), ("E4", 1)],
        [("F#4", .5), ("A4", .5), ("C#5", 1), ("B4", .5), ("A4", .5), ("B4", 1)],
    ]

    progression = [
        ("B1", ["B2", "F#3", "A3", "E4"]),
        ("G1", ["G2", "D3", "F#3", "B3"]),
        ("E1", ["E2", "B2", "D3", "F#3"]),
        ("F#1", ["F#2", "C#3", "E3", "G3"]),
        ("A1", ["A2", "E3", "F#3", "B3"]),
        ("D2", ["D3", "A3", "C#4", "F#4"]),
        ("E1", ["E2", "B2", "D3", "G3"]),
        ("F#1", ["F#2", "C#3", "E3", "A#3"]),
    ]

    def add_clav_bar(bar: int, chord_index: int, stronger: bool = False):
        start = bar * 4
        root, chord = progression[chord_index]
        # Syncopated but fixed eight-note groove, derived from chord tones.
        sequence = [chord[0], chord[1], chord[0], chord[2], chord[1], chord[3], chord[0], chord[1]]
        rests = {3} if not stronger else set()
        for step, pitch in enumerate(sequence):
            if step in rests:
                continue
            velocity = 70 if step % 4 == 0 else 55
            add_note(events, "Clav", start + step * .5, pitch, .28, velocity + (8 if stronger else 0))

    def add_bass_bar(bar: int, chord_index: int, dense: bool = False):
        start = bar * 4
        root, chord = progression[chord_index]
        notes = [root, chord[1], note_number(root) + 12, chord[2]]
        times = [0, 1.5, 2, 3.5] if dense else [0, 2]
        chosen = notes if dense else [root, note_number(root) + 12]
        for i, offset in enumerate(times):
            add_note(events, "Fretless Bass", start + offset, chosen[i], .72 if dense else 1.55, 88 if i == 0 else 74)

    def add_guitar_stabs(bar: int, chord_index: int, active: bool):
        if not active:
            return
        start = bar * 4
        _, chord = progression[chord_index]
        for offset in (0, 2.5):
            add_chord(events, "Power Guitar", start + offset,
                      [note_number(chord[0]) - 12, note_number(chord[1]) - 12], .30, 66)

    def add_drums(bar: int, energy: int):
        start = bar * 4
        kicks = [0, 1.5, 2.5] if energy == 0 else [0, .75, 1.5, 2.5, 3.5]
        snares = [1, 3]
        for offset in kicks:
            add_note(events, "Drums", start + offset, 36, .08, 95 if offset in (0, 2.5) else 78)
        for offset in snares:
            add_note(events, "Drums", start + offset, 38, .10, 108)
        for step in range(8):
            add_note(events, "Drums", start + step * .5, 42, .055, 46 if step % 2 else 60)
        if energy >= 1 and bar % 2 == 1:
            add_note(events, "Drums", start + 2.75, 40, .06, 68)
        if bar % 4 == 3:
            # One deliberate fill per phrase, not constant breakcore spam.
            for i, drum_note in enumerate([45, 47, 48, 50]):
                add_note(events, "Drums", start + 3 + i * .25, drum_note, .07, 82 + i * 5)
        if bar in (0, 8, 16, 24):
            add_note(events, "Drums", start, 49, .18, 104)

    # 0-7: exposed hook, baritone sax lead with the sans-like clav/bass rhythm.
    for bar in range(8):
        idx = bar
        start = bar * 4
        add_pattern(events, "Baritone Sax", start, hook[idx], 108)
        add_clav_bar(bar, idx, False)
        add_bass_bar(bar, idx, False)
        add_guitar_stabs(bar, idx, bar >= 4)
        add_drums(bar, 0)
        if idx in (0, 4):
            add_note(events, "Timpani", start, note_number(progression[idx][0]) + 12, .55, 80)

    # 8-15: hook restated by Romantic Trumpet; sax becomes a low answer.
    counter = [
        [("B3", 2), ("F#4", 2)], [("G3", 2), ("D4", 2)],
        [("E3", 2), ("B3", 2)], [("F#3", 2), ("C#4", 2)],
        [("A3", 2), ("E4", 2)], [("D3", 2), ("A3", 2)],
        [("E3", 2), ("B3", 2)], [("F#3", 2), ("C#4", 2)],
    ]
    for bar in range(8, 16):
        idx = bar - 8
        start = bar * 4
        add_pattern(events, "Romantic Trumpet", start, hook[idx], 110)
        add_pattern(events, "Baritone Sax", start, counter[idx], 58)
        add_clav_bar(bar, idx, True)
        add_bass_bar(bar, idx, True)
        add_guitar_stabs(bar, idx, True)
        add_drums(bar, 1)
        add_chord(events, "Strings", start, progression[idx][1], 3.75, 34)
        if idx in (0, 4):
            add_chord(events, "Orchestra Hit", start,
                      [progression[idx][1][0], progression[idx][1][2]], .18, 86)

    # 16-23: contrasting melody; sax and trumpet trade full phrases.
    for bar in range(16, 24):
        idx = bar - 16
        start = bar * 4
        lead = "Baritone Sax" if idx < 4 else "Romantic Trumpet"
        answer = "Romantic Trumpet" if idx < 4 else "Baritone Sax"
        add_pattern(events, lead, start, bridge[idx], 112)
        if idx % 2 == 1:
            add_pattern(events, answer, start, counter[idx], 55)
        add_clav_bar(bar, idx, True)
        add_bass_bar(bar, idx, True)
        add_guitar_stabs(bar, idx, True)
        add_drums(bar, 1)
        add_chord(events, "Strings", start, progression[idx][1], 3.75, 46)
        if idx in (3, 7):
            add_chord(events, "Orchestra Hit", start + 3.5,
                      [progression[idx][1][0], progression[idx][1][2], progression[idx][1][3]], .28, 108)

    # 24-31: final hook; both leads, choir and low pulse, no octave-up squeal.
    for bar in range(24, 32):
        idx = bar - 24
        start = bar * 4
        add_pattern(events, "Romantic Trumpet", start, hook[idx], 114)
        add_pattern(events, "Baritone Sax", start, hook[idx], 72, shift=-12)
        add_clav_bar(bar, idx, True)
        add_bass_bar(bar, idx, True)
        add_guitar_stabs(bar, idx, True)
        add_drums(bar, 1)
        add_chord(events, "Strings", start, progression[idx][1], 3.75, 52)
        if idx % 2 == 0:
            add_chord(events, "Choir Aahs", start, progression[idx][1], 3.75, 30)
        # Pulse acts as a low harmonic accent only.
        add_note(events, "25% Pulse", start, note_number(progression[idx][1][0]) + 12, 1.6, 46)
        add_note(events, "25% Pulse", start + 2, note_number(progression[idx][1][1]) + 12, 1.6, 42)
        if idx in (0, 4):
            add_note(events, "Timpani", start, note_number(progression[idx][0]) + 12, .62, 92)
        if idx in (3, 7):
            add_chord(events, "Orchestra Hit", start + 3.5,
                      [progression[idx][1][0], progression[idx][1][2], progression[idx][1][3]], .34, 118)

    return events


def render_sf2(soundfont: Path, preset, events: Sequence[Event], gain: float) -> np.ndarray:
    total_frames = int((BARS * BAR + TAIL) * SR)
    synth = fluidsynth.Synth(gain=gain, samplerate=SR)
    sfid = synth.sfload(str(soundfont))
    _, bank, program = preset
    channel = 9 if bank >= 120 else 0
    result = synth.program_select(channel, sfid, bank, program)
    if result != 0:
        synth.program_select(channel, sfid, 128 if channel == 9 else 0, program)
    synth.set_reverb(roomsize=.13, damping=.72, width=.68, level=.08)
    synth.set_chorus(nr=2, level=.10, speed=.22, depth=1.7, type=0)

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
            synth.noteon(channel, note, velocity)
        else:
            synth.noteoff(channel, note)
    if cursor < total_frames:
        chunks.append(synth.get_samples(total_frames - cursor))
    synth.delete()

    if not chunks:
        return np.zeros((total_frames, 2), dtype=np.float64)
    raw = np.concatenate(chunks).astype(np.float64) / 32768.0
    return raw.reshape(-1, 2)


def render_pulse(events: Sequence[Event]) -> np.ndarray:
    total_frames = int((BARS * BAR + TAIL) * SR)
    output = np.zeros((total_frames, 2), dtype=np.float64)
    for start, duration, note, velocity in events:
        seconds = duration * BEAT
        count = int((seconds + .04) * SR)
        time = np.arange(count) / SR
        phase = (note_hz(note) * time) % 1.0
        signal = np.where(phase < .25, 1.0, -1.0)
        envelope = np.minimum(1.0, time / .002) * (.30 + .70 * np.exp(-time / .25))
        gate = min(count, int(seconds * SR))
        if gate < count:
            envelope[gate:] *= np.linspace(1.0, 0.0, count - gate)
        signal = np.round(signal * 12.0) / 12.0
        signal *= envelope * (velocity / 127.0) * .075
        begin = int(start * BEAT * SR)
        end = min(total_frames, begin + count)
        output[begin:end, 0] += signal[:end - begin] * .70
        output[begin:end, 1] += signal[:end - begin] * .74
    return output


def write_wav(path: Path, audio: np.ndarray) -> None:
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak * .94
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes(pcm.tobytes())


def export_audio(wav_path: Path, mp3_path: Path, ogg_path: Path | None = None) -> None:
    effect = (
        "highpass=f=27,lowpass=f=11500,"
        "acompressor=threshold=-17dB:ratio=2.2:attack=8:release=105,"
        "alimiter=limit=0.96"
    )
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path),
        "-af", effect, "-codec:a", "libmp3lame", "-q:a", "2", str(mp3_path)
    ], check=True)
    if ogg_path is not None:
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path),
            "-af", effect, "-codec:a", "libvorbis", "-q:a", "6", str(ogg_path)
        ], check=True)


def export_midi(path: Path, events: Dict[str, List[Event]], preset_map) -> None:
    midi = MidiFile(ticks_per_beat=TPB)
    tempo_track = MidiTrack()
    tempo_track.append(MetaMessage("track_name", name="Tempo", time=0))
    tempo_track.append(MetaMessage("set_tempo", tempo=bpm2tempo(BPM), time=0))
    tempo_track.append(MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    midi.tracks.append(tempo_track)

    channel_cursor = 0
    for name, track_events in events.items():
        if not track_events:
            continue
        track = MidiTrack()
        track.append(MetaMessage("track_name", name=name, time=0))
        if name == "Drums":
            channel = 9
            _, bank, program = preset_map[name]
        elif name == "25% Pulse":
            channel, bank, program = 14, 0, 80
        else:
            while channel_cursor == 9:
                channel_cursor += 1
            channel = channel_cursor % 16
            channel_cursor += 1
            _, bank, program = preset_map[name]
        track.append(Message("control_change", channel=channel, control=0,
                             value=min(127, bank), time=0))
        track.append(Message("program_change", channel=channel,
                             program=min(127, program), time=0))
        timeline = []
        for start, duration, note, velocity in track_events:
            timeline.append((round(start * TPB), True, note, velocity))
            timeline.append((round((start + duration) * TPB), False, note, 0))
        timeline.sort(key=lambda item: (item[0], item[1]))
        previous = 0
        for tick, is_on, note, velocity in timeline:
            track.append(Message(
                "note_on" if is_on else "note_off", channel=channel, note=note,
                velocity=velocity if is_on else 0, time=tick - previous
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
    preset_map = build_preset_map(args.soundfont)

    selected = ["Exact presets selected from UNDERTALE Soundfont 2:"]
    for label, (name, bank, program) in preset_map.items():
        selected.append(f"{label}: {name} | bank={bank} preset={program}")
    (args.out / "KRIS_BAD_MEMORY_PRESETS.txt").write_text(
        "\n".join(selected) + "\n", encoding="utf-8"
    )

    gains = {
        "Clav": .56,
        "Baritone Sax": .82,
        "Romantic Trumpet": .83,
        "Fretless Bass": .62,
        "Power Guitar": .36,
        "Strings": .34,
        "Choir Aahs": .22,
        "Orchestra Hit": .38,
        "Timpani": .46,
        "Drums": .72,
    }

    total_frames = int((BARS * BAR + TAIL) * SR)
    full = np.zeros((total_frames, 2), dtype=np.float64)
    hook_proof = np.zeros_like(full)

    for label, gain in gains.items():
        stem = render_sf2(args.soundfont, preset_map[label], events[label], .78)
        length = min(total_frames, len(stem))
        full[:length] += stem[:length] * gain
        if label in ("Baritone Sax", "Clav", "Fretless Bass", "Drums"):
            hook_proof[:length] += stem[:length] * {
                "Baritone Sax": .92, "Clav": .58, "Fretless Bass": .62, "Drums": .60
            }[label]

    full += render_pulse(events["25% Pulse"])

    # Only the first eight bars in the proof file.
    hook_frames = int(8 * BAR * SR)
    hook_proof[hook_frames:] = 0
    full = np.tanh(full * 1.08)
    hook_proof = np.tanh(hook_proof * 1.05)

    full_wav = args.out / "KRIS_BAD_MEMORY_FULL.wav"
    hook_wav = args.out / "KRIS_BAD_MEMORY_HOOK.wav"
    write_wav(full_wav, full)
    write_wav(hook_wav, hook_proof[:hook_frames + int(.6 * SR)])

    export_audio(
        full_wav,
        args.out / "KRIS_BAD_MEMORY_FULL.mp3",
        args.out / "KRIS_BAD_MEMORY_GAME.ogg",
    )
    export_audio(hook_wav, args.out / "KRIS_BAD_MEMORY_HOOK.mp3")
    export_midi(args.out / "KRIS_BAD_MEMORY.mid", events, preset_map)

    print("\n".join(selected))
    print(f"Rendered at {BPM} BPM")


if __name__ == "__main__":
    main()
