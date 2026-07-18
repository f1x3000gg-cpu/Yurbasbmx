#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import subprocess
import wave
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import fluidsynth
import numpy as np
from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo
from sf2utils.sf2parse import Sf2File

SR = 44100
BPM = 150
BEAT = 60.0 / BPM
BAR_SECONDS = 4.0 * BEAT
BARS = 32
TPB = 480
TAIL_SECONDS = 2.2

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
                     duration * 0.94, velocity)
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
                drum_only: bool = False):
    rows = []
    for preset in presets:
        name = preset_name(preset)
        bank = preset_bank(preset)
        program = preset_program(preset)
        normalized = name.casefold().replace("_", " ").replace("-", " ")
        rows.append((normalized, name, bank, program))

    for keywords in alternatives:
        lowered = [word.casefold() for word in keywords]
        candidates = []
        for normalized, name, bank, program in rows:
            if drum_only and bank < 120:
                continue
            if all(word in normalized for word in lowered):
                score = len(normalized) + (0 if bank == 0 else 10)
                candidates.append((score, name, bank, program))
        if candidates:
            _, name, bank, program = sorted(candidates)[0]
            return name, bank, program

    return f"fallback bank {fallback_bank} program {fallback_program}", fallback_bank, fallback_program


def build_events() -> Dict[str, List[Event]]:
    names = [
        "Lead Trumpet", "Lead Violin", "Piano 1", "Strings", "Choir Aahs",
        "French Horns", "Orchestra Hit", "Timpani", "Glockenspiel",
        "Synth Bass", "Drums", "25% Pulse", "Triangle Bass",
    ]
    events: Dict[str, List[Event]] = {name: [] for name in names}

    # Eight-bar melody written first. Every bar is four beats and the line is
    # intentionally singable: rests, repeated anchor notes and longer endings.
    theme_a: List[Pattern] = [
        [(None, .5), ("B4", .5), ("E5", 1), ("F#5", .5), ("G5", .5), ("B4", 1)],
        [("D5", 1), ("B4", .5), ("A4", .5), ("G4", 1), ("B4", 1)],
        [(None, .5), ("C5", .5), ("E5", 1), ("D5", .5), ("B4", .5), ("A4", 1)],
        [("F#4", .5), ("A4", .5), ("B4", 1), ("D#5", .5), ("C5", .5), ("B4", 1)],
        [("B4", .5), ("E5", .5), ("G5", 1), ("F#5", .5), ("E5", .5), ("D5", 1)],
        [("C5", .5), ("B4", .5), ("G4", 1), ("A4", .5), ("C5", .5), ("B4", 1)],
        [("E5", 1), ("D5", .5), ("B4", .5), ("A4", 1), ("F#4", 1)],
        [("G4", .5), ("A4", .5), ("B4", 1), ("D#5", .5), ("E5", .5), ("E5", 1)],
    ]

    # A proper contrasting second theme, not random notes and not a transposed copy.
    theme_b: List[Pattern] = [
        [("E5", 1), ("G5", .5), ("F#5", .5), ("D5", 1), ("B4", 1)],
        [("C5", 1), ("E5", .5), ("D5", .5), ("B4", 1), ("G4", 1)],
        [("A4", .5), ("B4", .5), ("C5", 1), ("E5", .5), ("F#5", .5), ("G5", 1)],
        [("F#5", .5), ("E5", .5), ("D#5", 1), ("B4", 1), (None, 1)],
        [("G5", 1), ("E5", .5), ("D5", .5), ("B4", 1), ("C5", 1)],
        [("E5", .5), ("D5", .5), ("B4", 1), ("A4", .5), ("G4", .5), ("E4", 1)],
        [("F#4", .5), ("A4", .5), ("C5", 1), ("B4", .5), ("A4", .5), ("F#4", 1)],
        [("G4", .5), ("A4", .5), ("B4", 1), ("D#5", .5), ("E5", .5), ("E5", 1)],
    ]

    chords = [
        ["E3", "B3", "F#4", "G4"],
        ["D3", "G3", "B3", "F#4"],
        ["C3", "G3", "B3", "E4"],
        ["B2", "F#3", "A3", "D#4"],
        ["E3", "B3", "D4", "G4"],
        ["C3", "G3", "B3", "E4"],
        ["A2", "E3", "G3", "C4"],
        ["B2", "F#3", "A3", "D#4"],
    ]
    bass_roots = ["E2", "G2", "C2", "B1", "E2", "C2", "A1", "B1"]

    # First statement: melody is exposed and easy to judge.
    for bar in range(8):
        start = bar * 4
        add_pattern(events, "Lead Trumpet", start, theme_a[bar], 106)
        add_pattern(events, "Piano 1", start, theme_a[bar], 58, shift=-12)
        add_chord(events, "Strings", start, chords[bar], 3.78, 42)
        add_note(events, "Triangle Bass", start, bass_roots[bar], 1.8, 92)
        add_note(events, "Triangle Bass", start + 2, bass_roots[bar], 1.8, 78)

    # Second statement: same identity, fuller harmony and a written counter-line.
    long_counter = [
        [("E4", 2), ("B3", 2)], [("D4", 2), ("B3", 2)],
        [("C4", 2), ("G3", 2)], [("B3", 2), ("F#3", 2)],
        [("G4", 2), ("E4", 2)], [("E4", 2), ("C4", 2)],
        [("C4", 2), ("A3", 2)], [("D#4", 2), ("B3", 2)],
    ]
    for bar in range(8, 16):
        idx = bar - 8
        start = bar * 4
        add_pattern(events, "Lead Violin", start, theme_a[idx], 104)
        add_pattern(events, "Lead Trumpet", start, long_counter[idx], 58)
        add_chord(events, "Strings", start, chords[idx], 3.78, 54)
        if idx % 2 == 0:
            add_chord(events, "Choir Aahs", start, chords[idx], 3.75, 34)
        add_note(events, "Synth Bass", start, bass_roots[idx], 1.7, 84)
        add_note(events, "Synth Bass", start + 2, bass_roots[idx], 1.7, 76)
        if idx in (0, 4):
            add_chord(events, "French Horns", start, [chords[idx][0], chords[idx][2]], 1.4, 68)

    # Contrasting B section.
    for bar in range(16, 24):
        idx = bar - 16
        start = bar * 4
        add_pattern(events, "Lead Trumpet", start, theme_b[idx], 110)
        add_pattern(events, "Piano 1", start, theme_b[idx], 62, shift=-12)
        add_chord(events, "Strings", start, chords[idx], 3.8, 58)
        add_note(events, "Synth Bass", start, bass_roots[idx], .9, 88)
        add_note(events, "Synth Bass", start + 1.5, bass_roots[idx], .4, 68)
        add_note(events, "Synth Bass", start + 2, bass_roots[idx], .9, 80)
        add_note(events, "Synth Bass", start + 3.5, bass_roots[idx], .35, 66)
        if idx in (3, 7):
            add_chord(events, "Orchestra Hit", start + 3.5,
                      [chords[idx][0], chords[idx][2], chords[idx][3]], .32, 104)

    # Final statement: A returns with a restrained low pulse and countervoice.
    for bar in range(24, 32):
        idx = bar - 24
        start = bar * 4
        add_pattern(events, "Lead Violin", start, theme_a[idx], 114)
        add_pattern(events, "Lead Trumpet", start, long_counter[idx], 72)
        add_chord(events, "Strings", start, chords[idx], 3.8, 64)
        add_chord(events, "Choir Aahs", start, chords[idx], 3.8, 42)
        add_note(events, "Synth Bass", start, bass_roots[idx], 1.75, 92)
        add_note(events, "Synth Bass", start + 2, bass_roots[idx], 1.75, 82)
        # The pulse stays below the melody and plays a written half-note rhythm.
        add_note(events, "25% Pulse", start, note_number(chords[idx][0]) + 12, 1.75, 54)
        add_note(events, "25% Pulse", start + 2, note_number(chords[idx][2]) + 12, 1.75, 50)
        if idx in (0, 4):
            add_chord(events, "French Horns", start,
                      [chords[idx][0], chords[idx][2]], 1.5, 76)
        if idx in (3, 7):
            add_chord(events, "Orchestra Hit", start + 3.5,
                      [chords[idx][0], chords[idx][2], chords[idx][3]], .38, 116)

    # Sparse glockenspiel only as punctuation, never as a shrill lead.
    for beat, pitch in [(31.5, "B5"), (63.5, "E6"), (95.5, "B5"), (127.5, "E6")]:
        add_note(events, "Glockenspiel", beat, pitch, .25, 42)

    # Timpani phrase markers.
    for bar in (0, 8, 16, 24, 28):
        root = bass_roots[bar % 8]
        add_note(events, "Timpani", bar * 4, note_number(root) + 12, .65, 86)

    # Drum writing: energetic but not a continuous wall of 32nd notes.
    for bar in range(BARS):
        start = bar * 4
        section = bar // 8
        # Backbeat foundation.
        add_note(events, "Drums", start, 36, .10, 100)
        add_note(events, "Drums", start + 1, 38, .11, 108)
        add_note(events, "Drums", start + 2, 36, .10, 96)
        add_note(events, "Drums", start + 3, 38, .11, 112)
        for step in range(8):
            add_note(events, "Drums", start + step * .5, 42, .06,
                     46 if step % 2 else 58)
        if section >= 1:
            add_note(events, "Drums", start + 1.5, 36, .08, 76)
            add_note(events, "Drums", start + 3.5, 36, .08, 80)
        if section >= 2 and bar % 2 == 1:
            add_note(events, "Drums", start + 2.5, 40, .07, 68)
            add_note(events, "Drums", start + 2.75, 38, .07, 78)
        # One deliberate breakcore fill at each four-bar cadence.
        if section >= 2 and bar % 4 == 3:
            fill = [(3.0, 45), (3.25, 47), (3.5, 48), (3.75, 50)]
            for offset, drum_note in fill:
                add_note(events, "Drums", start + offset, drum_note, .07, 84)
        if bar in (0, 8, 16, 24):
            add_note(events, "Drums", start, 49, .20, 104)

    return events


def build_preset_map(soundfont_path: Path):
    with soundfont_path.open("rb") as handle:
        sf2 = Sf2File(handle)
        presets = list(sf2.presets)

    mapping = {
        "Lead Trumpet": find_preset(
            presets,
            [["romantic", "tp"], ["romantic", "trumpet"], ["solo", "trumpet"], ["trumpet"]],
            0, 56,
        ),
        "Lead Violin": find_preset(
            presets,
            [["thfont", "violin"], ["fast", "violin"], ["violin"]],
            0, 40,
        ),
        "Piano 1": find_preset(presets, [["piano", "1"], ["piano"]], 0, 0),
        "Strings": find_preset(presets, [["sgm", "strings"], ["strings"]], 0, 48),
        "Choir Aahs": find_preset(presets, [["choir", "aahs"], ["choir"]], 0, 52),
        "French Horns": find_preset(presets, [["french", "horn"], ["horn"]], 0, 60),
        "Orchestra Hit": find_preset(
            presets,
            [["orchestra", "hit"], ["orchestrahit"], ["impact", "hit"]],
            0, 55,
        ),
        "Timpani": find_preset(presets, [["timpani"]], 0, 47),
        "Glockenspiel": find_preset(presets, [["glockenspiel"], ["glock"]], 0, 9),
        "Synth Bass": find_preset(
            presets,
            [["synth", "bass", "2"], ["synth", "bass"], ["fingered", "bass"]],
            0, 39,
        ),
        "Drums": find_preset(
            presets,
            [["power"], ["standard", "1"], ["standard"]],
            128, 16, drum_only=True,
        ),
    }
    return mapping


def render_soundfont_events(soundfont_path: Path, preset, events: Sequence[Event],
                            gain: float = .8) -> np.ndarray:
    total_frames = int((BARS * BAR_SECONDS + TAIL_SECONDS) * SR)
    synth = fluidsynth.Synth(gain=gain, samplerate=SR)
    sfid = synth.sfload(str(soundfont_path))
    _, bank, program = preset
    channel = 9 if bank >= 120 else 0
    result = synth.program_select(channel, sfid, bank, program)
    if result != 0:
        # The exact SoundFont is still loaded; this fallback selects its GM slot.
        synth.program_select(channel, sfid, 128 if channel == 9 else 0, program)
    synth.set_reverb(roomsize=.16, damping=.68, width=.72, level=.10)
    synth.set_chorus(nr=2, level=.16, speed=.25, depth=2.0, type=0)

    timeline = []
    for start, duration, note, velocity in events:
        on = int(start * BEAT * SR)
        off = int((start + duration) * BEAT * SR)
        timeline.append((on, True, note, velocity))
        timeline.append((off, False, note, 0))
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
    total_frames = int((BARS * BAR_SECONDS + TAIL_SECONDS) * SR)
    stereo = np.zeros((total_frames, 2), dtype=np.float64)
    for start, duration, note, velocity in events:
        seconds = duration * BEAT
        count = int((seconds + .05) * SR)
        time = np.arange(count) / SR
        frequency = note_hz(note)
        phase = (frequency * time) % 1.0
        signal = np.where(phase < .25, 1.0, -1.0)
        envelope = np.minimum(1.0, time / .002) * (.28 + .72 * np.exp(-time / .28))
        gate = min(count, int(seconds * SR))
        if gate < count:
            envelope[gate:] *= np.linspace(1.0, 0.0, count - gate)
        signal = np.round(signal * 12.0) / 12.0
        signal *= envelope * (velocity / 127.0) * .09
        begin = int(start * BEAT * SR)
        end = min(total_frames, begin + count)
        stereo[begin:end, 0] += signal[:end - begin] * .68
        stereo[begin:end, 1] += signal[:end - begin] * .74
    return stereo


def render_triangle(events: Sequence[Event]) -> np.ndarray:
    total_frames = int((BARS * BAR_SECONDS + TAIL_SECONDS) * SR)
    stereo = np.zeros((total_frames, 2), dtype=np.float64)
    for start, duration, note, velocity in events:
        seconds = duration * BEAT
        count = int((seconds + .04) * SR)
        time = np.arange(count) / SR
        frequency = note_hz(note)
        phase = (frequency * time) % 1.0
        signal = 4.0 * np.abs(phase - .5) - 1.0
        signal = np.round(signal * 16.0) / 16.0
        envelope = np.minimum(1.0, time / .003) * (.22 + .78 * np.exp(-time / .36))
        gate = min(count, int(seconds * SR))
        if gate < count:
            envelope[gate:] *= np.linspace(1.0, 0.0, count - gate)
        signal *= envelope * (velocity / 127.0) * .15
        begin = int(start * BEAT * SR)
        end = min(total_frames, begin + count)
        stereo[begin:end, 0] += signal[:end - begin] * .72
        stereo[begin:end, 1] += signal[:end - begin] * .70
    return stereo


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


def export_audio(wav_path: Path, mp3_path: Path, ogg_path: Path | None = None) -> None:
    audio_filter = (
        "highpass=f=28,lowpass=f=12000,"
        "acompressor=threshold=-18dB:ratio=2.1:attack=10:release=120,"
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


def export_midi(path: Path, events: Dict[str, List[Event]], preset_map) -> None:
    midi = MidiFile(ticks_per_beat=TPB)
    tempo_track = MidiTrack()
    tempo_track.append(MetaMessage("track_name", name="Tempo", time=0))
    tempo_track.append(MetaMessage("set_tempo", tempo=bpm2tempo(BPM), time=0))
    tempo_track.append(MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    midi.tracks.append(tempo_track)

    channel = 0
    for track_name, track_events in events.items():
        if not track_events:
            continue
        track = MidiTrack()
        track.append(MetaMessage("track_name", name=track_name, time=0))
        if track_name == "Drums":
            midi_channel = 9
            _, bank, program = preset_map[track_name]
        elif track_name in ("25% Pulse", "Triangle Bass"):
            midi_channel = 14 if track_name == "25% Pulse" else 15
            bank, program = 0, 80 if track_name == "25% Pulse" else 38
        else:
            while channel == 9:
                channel += 1
            midi_channel = channel % 16
            channel += 1
            _, bank, program = preset_map[track_name]
        track.append(Message("control_change", channel=midi_channel, control=0,
                             value=min(127, bank), time=0))
        track.append(Message("program_change", channel=midi_channel,
                             program=min(127, program), time=0))
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
    preset_map = build_preset_map(args.soundfont)

    selected_text = ["Selected presets from UNDERTALE Soundfont 2:"]
    for label, (name, bank, program) in preset_map.items():
        selected_text.append(f"{label}: {name} | bank={bank} preset={program}")
    (args.out / "SELECTED_UNDERTALE_PRESETS.txt").write_text(
        "\n".join(selected_text) + "\n", encoding="utf-8"
    )

    stem_gains = {
        "Lead Trumpet": .88,
        "Lead Violin": .86,
        "Piano 1": .56,
        "Strings": .48,
        "Choir Aahs": .34,
        "French Horns": .45,
        "Orchestra Hit": .45,
        "Timpani": .52,
        "Glockenspiel": .20,
        "Synth Bass": .62,
        "Drums": .66,
    }

    total_frames = int((BARS * BAR_SECONDS + TAIL_SECONDS) * SR)
    full = np.zeros((total_frames, 2), dtype=np.float64)
    melody_only = np.zeros_like(full)

    for label, gain in stem_gains.items():
        stem = render_soundfont_events(args.soundfont, preset_map[label], events[label])
        length = min(len(full), len(stem))
        full[:length] += stem[:length] * gain
        if label in ("Lead Trumpet", "Lead Violin", "Piano 1"):
            melody_only[:length] += stem[:length] * ({
                "Lead Trumpet": .95, "Lead Violin": .92, "Piano 1": .58
            }[label])

    full += render_pulse(events["25% Pulse"])
    full += render_triangle(events["Triangle Bass"])

    # Mild saturation only; the melody must remain the foreground.
    full = np.tanh(full * 1.08)
    melody_only = np.tanh(melody_only * 1.03)

    full_wav = args.out / "KRIS_MELODY_FIRST_FULL.wav"
    melody_wav = args.out / "KRIS_MELODY_FIRST_8BAR_PROOF.wav"
    write_wav(full_wav, full)
    write_wav(melody_wav, melody_only)

    export_audio(
        full_wav,
        args.out / "KRIS_MELODY_FIRST_FULL.mp3",
        args.out / "KRIS_MELODY_FIRST_GAME.ogg",
    )
    export_audio(
        melody_wav,
        args.out / "KRIS_MELODY_FIRST_8BAR_PROOF.mp3",
    )
    export_midi(args.out / "KRIS_MELODY_FIRST.mid", events, preset_map)

    print("\n".join(selected_text))
    print(f"Rendered at {BPM} BPM")


if __name__ == "__main__":
    main()
