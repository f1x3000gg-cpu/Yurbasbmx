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
BPM = 156
BEAT = 60.0 / BPM
BAR_SECONDS = 4.0 * BEAT
BARS = 36
TAIL_SECONDS = 2.4
TPB = 480

PITCH = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

Event = Tuple[float, float, int, int]
Pattern = Sequence[Tuple[str | int | None, float]]

# Serious orchestral palette taken from named popular UNDERTALE track groups.
# There is no saxophone, guitar, generic GM fallback, or random preset search.
PRESETS = {
    "Spear Orchestra Hit": (1, 58, "046 Spear of Justice - Orchestra Hit"),
    "Spear Strings":       (1, 60, "046 Spear of Justice - Strings"),
    "Spear Piano":         (1, 61, "046 Spear of Justice - Piano"),
    "Core Choir":          (2, 32, "065 CORE - Ahh Choir"),
    "Core Brass":          (2, 35, "065 CORE - Brass"),
    "Asgore Piano":        (2, 64, "077 ASGORE - Piano"),
    "Asgore Timpani":      (2, 74, "077 ASGORE - Timpani"),
    "Asgore Strings":      (2, 75, "077 ASGORE - Strings"),
    "Asgore Choir":        (2, 76, "077 ASGORE - Choir Aahs"),
    "Asgore Brass":        (2, 78, "077 ASGORE - Brass"),
    "Asgore Drums":        (2, 80, "077 ASGORE - Drumkit"),
    "Finale Piano":        (2, 93, "080 Finale - Piano 1"),
    "Finale Strings":      (2, 97, "080 Finale - Strings"),
    "Finale Trumpet":      (2, 98, "080 Finale - Trumpet"),
    "Finale Marcato":      (2, 99, "080 Finale - Strings marc"),
    "Hopes Violin":        (2, 125, "087 Hopes and Dreams - Violin Detache"),
    "Hopes Power Drums":   (2, 127, "087 Hopes and Dreams - POWER DrumKit"),
    "Hero Romantic Trumpet": (3, 65, "098 Battle Against a True Hero - Romantic Tp"),
    "Hero Choir":          (3, 68, "098 Battle Against a True Hero - Choir Aahs"),
    "Hero Strings":        (3, 72, "098 Battle Against a True Hero - Strings"),
    "Neo Organ":           (3, 76, "099 Power of NEO - Organ 3"),
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

    # C minor. Every phrase was written by hand before orchestration.
    # The melodic rhythm is mostly quarter/eighth notes with rests and long
    # cadence tones, so the theme can be sung instead of sounding generated.
    theme_a: List[Pattern] = [
        [(None,.5),("G4",.5),("C5",1),("Eb5",1),("D5",1)],
        [("Bb4",.5),("G4",.5),("F4",1),("G4",1),("C5",1)],
        [(None,.5),("Eb5",.5),("G5",1),("F5",.5),("Eb5",.5),("D5",1)],
        [("C5",1),("Bb4",.5),("G4",.5),("B4",1),("C5",1)],
        [("G4",.5),("C5",.5),("Eb5",1),("G5",1),("F5",1)],
        [("Eb5",.5),("D5",.5),("C5",1),("Bb4",1),("G4",1)],
        [("Ab4",.5),("C5",.5),("F5",1),("Eb5",.5),("D5",.5),("C5",1)],
        [("B4",.5),("C5",.5),("D5",1),("F5",.5),("Eb5",.5),("C5",1)],
    ]

    answer_b: List[Pattern] = [
        [("C5",1),("Eb5",.5),("D5",.5),("C5",1),("G4",1)],
        [("Ab4",.5),("Bb4",.5),("C5",1),("Eb5",1),("D5",1)],
        [("F5",.5),("Eb5",.5),("D5",1),("C5",.5),("Bb4",.5),("G4",1)],
        [("B4",.5),("C5",.5),("D5",1),("G5",1),("F5",1)],
        [("Eb5",.5),("G5",.5),("Ab5",1),("G5",.5),("F5",.5),("Eb5",1)],
        [("D5",.5),("C5",.5),("Bb4",1),("G4",.5),("Bb4",.5),("C5",1)],
        [("F5",1),("Eb5",.5),("D5",.5),("C5",1),("G4",1)],
        [("Ab4",.5),("B4",.5),("C5",1),("D5",.5),("B4",.5),("C5",1)],
    ]

    chorus: List[Pattern] = [
        [("C5",.5),("Eb5",.5),("G5",1),("Ab5",1),("G5",1)],
        [("F5",.5),("Eb5",.5),("D5",1),("C5",2)],
        [("Eb5",.5),("G5",.5),("Bb5",1),("Ab5",.5),("G5",.5),("F5",1)],
        [("G5",.5),("F5",.5),("Eb5",1),("D5",1),("C5",1)],
        [("G4",.5),("C5",.5),("Eb5",1),("F5",.5),("G5",.5),("Ab5",1)],
        [("G5",.5),("F5",.5),("Eb5",1),("D5",.5),("C5",.5),("Bb4",1)],
        [("Ab4",.5),("C5",.5),("F5",1),("Eb5",.5),("D5",.5),("C5",1)],
        [("C5",2),("B4",.5),("D5",.5),("C5",1)],
    ]

    bridge: List[Pattern] = [
        [("C5",2),("Eb5",1),("D5",1)],
        [("Ab4",2),("C5",1),("Eb5",1)],
        [("F5",1),("Eb5",1),("D5",2)],
        [("B4",1),("D5",1),("C5",2)],
    ]

    chords = [
        ["C3","G3","Bb3","Eb4"],
        ["Ab2","Eb3","G3","C4"],
        ["Eb3","Bb3","D4","G4"],
        ["G2","D3","F3","B3"],
        ["F2","C3","Eb3","Ab3"],
        ["Ab2","Eb3","G3","C4"],
        ["G2","D3","F3","B3"],
        ["G2","D3","F3","B3"],
    ]
    roots = ["C2","Ab1","Eb2","G1","F2","Ab1","G1","G1"]

    def bass_bar(bar: int, active: bool) -> None:
        start = bar * 4
        root = note_number(roots[bar % 8])
        if active:
            pattern = [root, root+7, root+12, root+7]
            for step, pitch in enumerate(pattern):
                add_note(events, "Neo Organ", start + step, pitch, .76,
                         62 if step else 78)
        else:
            add_note(events, "Neo Organ", start, root, 1.75, 66)
            add_note(events, "Neo Organ", start+2, root, 1.75, 58)

    def piano_chords(bar: int, force: int = 56) -> None:
        start = bar * 4
        chord = chords[bar % 8]
        add_chord(events, "Asgore Piano", start, chord, .52, force)
        add_chord(events, "Asgore Piano", start+2, chord, .52, force-5)

    def drums(bar: int, intensity: int, half_time: bool = False,
              fill: bool = False) -> None:
        start = bar * 4
        drum_track = "Hopes Power Drums" if intensity >= 2 else "Asgore Drums"
        if half_time:
            for pos in (0, 2.5):
                add_note(events, drum_track, start+pos, 36, .09, 94)
            add_note(events, drum_track, start+2, 38, .11, 112)
            for step in range(8):
                add_note(events, drum_track, start+step*.5, 42, .055,
                         42 if step % 2 else 52)
        else:
            for pos in (0, 2):
                add_note(events, drum_track, start+pos, 36, .09, 94+intensity*5)
            if intensity >= 1:
                add_note(events, drum_track, start+1.5, 36, .07, 72)
                add_note(events, drum_track, start+3.5, 36, .07, 76)
            for pos in (1, 3):
                add_note(events, drum_track, start+pos, 38, .10, 104+intensity*4)
            for step in range(8):
                add_note(events, drum_track, start+step*.5, 42, .055,
                         44 if step % 2 else 57)
        if fill:
            for offset, drum_note in ((3.0,45),(3.25,47),(3.5,48),(3.75,50)):
                add_note(events, drum_track, start+offset, drum_note, .07, 80)

    # 0-7: immediate serious statement. The lead is exposed enough that the
    # melody can be judged, but the battle energy starts from the first bar.
    for bar in range(8):
        start = bar * 4
        idx = bar
        add_pattern(events, "Hero Romantic Trumpet", start, theme_a[idx], 106)
        add_pattern(events, "Asgore Piano", start, theme_a[idx], 54, shift=-12)
        add_chord(events, "Asgore Strings", start, chords[idx], 3.78, 46)
        piano_chords(bar, 48)
        bass_bar(bar, False)
        drums(bar, 0, fill=(bar == 7))
        if bar in (0,4):
            add_chord(events, "Spear Orchestra Hit", start,
                      [chords[idx][0],chords[idx][2],chords[idx][3]], .22, 92)
            add_note(events, "Asgore Timpani", start, note_number(roots[idx])+12, .58, 88)

    # 8-15: a genuinely different answer phrase, led by violin and trumpet.
    for bar in range(8,16):
        start = bar * 4
        idx = bar-8
        add_pattern(events, "Hopes Violin", start, answer_b[idx], 108)
        add_pattern(events, "Finale Trumpet", start, answer_b[idx], 68, shift=-12)
        add_chord(events, "Finale Strings", start, chords[idx], 3.78, 54)
        add_chord(events, "Core Choir", start, chords[idx], 3.78, 28)
        piano_chords(bar, 54)
        bass_bar(bar, True)
        drums(bar, 1, fill=(idx in (3,7)))
        if idx in (0,4):
            add_note(events, "Asgore Timpani", start, note_number(roots[idx])+12, .62, 94)

    # 16-23: broad, memorable chorus. Brass supports the melody instead of
    # replacing it with fast notes.
    for bar in range(16,24):
        start = bar * 4
        idx = bar-16
        add_pattern(events, "Hero Romantic Trumpet", start, chorus[idx], 114)
        add_pattern(events, "Hopes Violin", start, chorus[idx], 88)
        add_chord(events, "Hero Strings", start, chords[idx], 3.78, 62)
        add_chord(events, "Hero Choir", start, chords[idx], 3.78, 40)
        add_pattern(events, "Core Brass", start,
                    [(chords[idx][0],2),(chords[idx][2],2)], 58)
        piano_chords(bar, 60)
        bass_bar(bar, True)
        drums(bar, 2, fill=(idx in (3,7)))
        if idx in (0,4):
            add_chord(events, "Spear Orchestra Hit", start,
                      [chords[idx][0],chords[idx][2],chords[idx][3]], .24, 108)
            add_note(events, "Asgore Timpani", start, note_number(roots[idx])+12, .68, 102)

    # 24-27: serious half-time bridge with long notes and no melodic clutter.
    for bar in range(24,28):
        start = bar * 4
        idx = bar-24
        chord_idx = idx+4
        add_pattern(events, "Finale Piano", start, bridge[idx], 100)
        add_pattern(events, "Hopes Violin", start, bridge[idx], 64, shift=-12)
        add_chord(events, "Asgore Strings", start, chords[chord_idx], 3.8, 62)
        add_chord(events, "Asgore Choir", start, chords[chord_idx], 3.8, 38)
        bass_bar(bar, False)
        drums(bar, 1, half_time=True, fill=(bar == 27))
        add_note(events, "Asgore Timpani", start, note_number(roots[chord_idx])+12, .62, 84)

    # 28-35: final chorus with the same hook made larger, not rewritten into
    # random high notes. Marcato strings provide aggression without guitars.
    for bar in range(28,36):
        start = bar * 4
        idx = bar-28
        add_pattern(events, "Hero Romantic Trumpet", start, chorus[idx], 120)
        add_pattern(events, "Hopes Violin", start, chorus[idx], 98)
        add_pattern(events, "Finale Trumpet", start, chorus[idx], 72, shift=-12)
        add_chord(events, "Finale Strings", start, chords[idx], 3.78, 68)
        add_chord(events, "Hero Choir", start, chords[idx], 3.78, 46)
        for beat in (0,1,2,3):
            add_chord(events, "Finale Marcato", start+beat,
                      [chords[idx][0],chords[idx][2]], .30, 60 if beat % 2 else 72)
        add_pattern(events, "Asgore Brass", start,
                    [(chords[idx][0],2),(chords[idx][3],2)], 64)
        piano_chords(bar, 64)
        bass_bar(bar, True)
        drums(bar, 3, fill=(idx in (1,3,5,7)))
        if idx in (0,4):
            add_chord(events, "Spear Orchestra Hit", start,
                      [chords[idx][0],chords[idx][2],chords[idx][3]], .26, 118)
            add_note(events, "Asgore Timpani", start, note_number(roots[idx])+12, .72, 112)

    # Final C-minor statement.
    end = BARS * 4 - .5
    final_chord = ["C2","G2","C3","Eb3","G3","C4"]
    add_chord(events, "Spear Orchestra Hit", end, final_chord, .42, 127)
    add_chord(events, "Asgore Piano", end, final_chord, .62, 118)
    add_chord(events, "Finale Strings", end, final_chord, .68, 108)
    add_note(events, "Asgore Timpani", end, "C2", .64, 127)
    add_note(events, "Hopes Power Drums", end, 49, .30, 127)

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
    synth.set_reverb(roomsize=.17, damping=.66, width=.74, level=.10)
    synth.set_chorus(nr=2, level=.12, speed=.25, depth=1.8, type=0)

    timeline = []
    for start, duration, note, velocity in events:
        timeline.append((int(start * BEAT * SR), True, note, velocity))
        timeline.append((int((start + duration) * BEAT * SR), False, note, 0))
    timeline.sort(key=lambda item: (item[0], item[1]))

    chunks = []
    cursor = 0
    for frame, is_on, note, velocity in timeline:
        if frame > cursor:
            chunks.append(synth.get_samples(frame-cursor))
            cursor = frame
        if is_on:
            synth.noteon(0, note, velocity)
        else:
            synth.noteoff(0, note)
    if cursor < total:
        chunks.append(synth.get_samples(total-cursor))
    synth.delete()

    if not chunks:
        return np.zeros((total,2), dtype=np.float64)
    raw = np.concatenate(chunks).astype(np.float64) / 32768.0
    return raw.reshape(-1,2)


def write_wav(path: Path, audio: np.ndarray) -> None:
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak * .94
    pcm = (np.clip(audio,-1,1)*32767).astype(np.int16)
    with wave.open(str(path),"wb") as handle:
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
    ],check=True)
    if ogg_path is not None:
        subprocess.run([
            "ffmpeg","-y","-loglevel","error","-i",str(wav_path),
            "-af",filt,"-codec:a","libvorbis","-q:a","6",str(ogg_path)
        ],check=True)


def export_midi(path: Path, events: Dict[str,List[Event]]) -> None:
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
        bank,preset,_ = PRESETS[name]
        while channel == 9:
            channel += 1
        midi_channel = channel % 16
        channel += 1
        track.append(Message("control_change", channel=midi_channel, control=0,
                             value=min(127,bank), time=0))
        track.append(Message("program_change", channel=midi_channel,
                             program=min(127,preset), time=0))
        timeline=[]
        for start,duration,note,velocity in track_events:
            timeline.append((round(start*TPB),True,note,velocity))
            timeline.append((round((start+duration)*TPB),False,note,0))
        timeline.sort(key=lambda item:(item[0],item[1]))
        previous=0
        for tick,is_on,note,velocity in timeline:
            track.append(Message(
                "note_on" if is_on else "note_off",
                channel=midi_channel,note=note,
                velocity=velocity if is_on else 0,
                time=tick-previous,
            ))
            previous=tick
        track.append(MetaMessage("end_of_track",time=1))
        midi.tracks.append(track)
    midi.save(path)


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--soundfont",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)

    events=build_events()
    total=int((BARS*BAR_SECONDS+TAIL_SECONDS)*SR)
    full=np.zeros((total,2),dtype=np.float64)
    proof=np.zeros_like(full)

    gains={
        "Spear Orchestra Hit":.48,
        "Spear Strings":.0,
        "Spear Piano":.0,
        "Core Choir":.26,
        "Core Brass":.38,
        "Asgore Piano":.58,
        "Asgore Timpani":.54,
        "Asgore Strings":.52,
        "Asgore Choir":.30,
        "Asgore Brass":.42,
        "Asgore Drums":.60,
        "Finale Piano":.58,
        "Finale Strings":.56,
        "Finale Trumpet":.50,
        "Finale Marcato":.42,
        "Hopes Violin":.70,
        "Hopes Power Drums":.70,
        "Hero Romantic Trumpet":.82,
        "Hero Choir":.34,
        "Hero Strings":.54,
        "Neo Organ":.40,
    }
    proof_tracks={
        "Hero Romantic Trumpet":.94,
        "Hopes Violin":.82,
        "Asgore Piano":.58,
        "Asgore Strings":.42,
        "Core Choir":.18,
    }

    selected=[
        f"BPM: {BPM}",
        "Key: C minor",
        "No saxophone. No guitar. No generic GM fallback.",
        "Melody written by hand as theme A, answer B, chorus, bridge, reprise.",
        "",
    ]

    for name,track_events in events.items():
        if not track_events:
            continue
        bank,preset,source=PRESETS[name]
        selected.append(f"{name}: bank={bank} preset={preset} | {source}")
        stem=render_stem(args.soundfont,bank,preset,track_events)
        length=min(total,len(stem))
        gain=gains.get(name,0.0)
        if gain:
            full[:length]+=stem[:length]*gain
        if name in proof_tracks:
            proof_length=min(length,int((24*BAR_SECONDS+1.0)*SR))
            proof[:proof_length]+=stem[:proof_length]*proof_tracks[name]

    full=np.tanh(full*1.05)
    proof=np.tanh(proof*1.02)
    proof=proof[:int((24*BAR_SECONDS+1.2)*SR)]

    full_wav=args.out/"KRIS_SERIOUS_THEME_FULL.wav"
    proof_wav=args.out/"KRIS_SERIOUS_THEME_MELODY.wav"
    write_wav(full_wav,full)
    write_wav(proof_wav,proof)
    encode(full_wav,args.out/"KRIS_SERIOUS_THEME_FULL.mp3",
           args.out/"KRIS_SERIOUS_THEME_GAME.ogg")
    encode(proof_wav,args.out/"KRIS_SERIOUS_THEME_MELODY.mp3")
    export_midi(args.out/"KRIS_SERIOUS_THEME.mid",events)
    (args.out/"KRIS_SERIOUS_THEME_INSTRUMENTS.txt").write_text(
        "\n".join(selected)+"\n",encoding="utf-8"
    )
    print("\n".join(selected))


if __name__=="__main__":
    main()
