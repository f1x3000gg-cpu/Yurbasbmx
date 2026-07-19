#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import wave
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import fluidsynth
import numpy as np
from sf2utils.sf2parse import Sf2File

SR = 44100
BPM = 166
BEAT = 60.0 / BPM
BAR = 4.0 * BEAT
BARS = 32
TAIL = 2.0

PITCH = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

Event = Tuple[float, float, int, int]
Pattern = Sequence[Tuple[str | int | None, float]]


def nn(note: str | int) -> int:
    if isinstance(note, int):
        return note
    if len(note) >= 3 and note[1] in "#b":
        name, octave = note[:2], int(note[2:])
    else:
        name, octave = note[0], int(note[1:])
    return 12 * (octave + 1) + PITCH[name]


def add(events: Dict[str, List[Event]], track: str, start: float,
        note: str | int, duration: float, velocity: int) -> None:
    events[track].append((start, duration, nn(note), velocity))


def pattern(events: Dict[str, List[Event]], track: str, start: float,
            notes: Pattern, velocity: int, shift: int = 0) -> None:
    cursor = start
    for note, duration in notes:
        if note is not None:
            add(events, track, cursor, nn(note) + shift, duration * .93, velocity)
        cursor += duration


def chord(events: Dict[str, List[Event]], track: str, start: float,
          notes: Sequence[str | int], duration: float, velocity: int) -> None:
    for note in notes:
        add(events, track, start, note, duration, velocity)


def preset_text(preset) -> str:
    value = getattr(preset, "name", "")
    if isinstance(value, bytes):
        return value.decode("latin1", errors="replace")
    return str(value)


def find_named_preset(path: Path, alternatives: Sequence[Sequence[str]]):
    with path.open("rb") as handle:
        presets = list(Sf2File(handle).presets)
    for words in alternatives:
        lowered = [word.casefold() for word in words]
        candidates = []
        for preset in presets:
            name = preset_text(preset)
            normalized = name.casefold().replace("_", " ").replace("-", " ")
            if all(word in normalized for word in lowered):
                candidates.append((len(normalized), name, int(preset.bank), int(preset.preset)))
        if candidates:
            _, name, bank, program = sorted(candidates)[0]
            return name, bank, program
    raise RuntimeError(f"Required preset not found: {alternatives}")


def build_events() -> Dict[str, List[Event]]:
    names = [
        "Comp Clav", "Baritone Sax", "Romantic Trumpet", "Fretless Bass",
        "Power Guitar", "Strings", "Choir Aahs", "Orchestra Hit",
        "Timpani", "POWER Drums", "25% Pulse",
    ]
    events: Dict[str, List[Event]] = {name: [] for name in names}

    # Original eight-bar melody: question, expansion, answer, cadence.
    hook: List[Pattern] = [
        [(None,.5),("F#4",.5),("B4",.5),("C#5",.5),("D5",1),("A4",.5),("F#4",.5)],
        [("E4",.5),("F#4",.5),("A4",1),("G4",.5),("E4",.5),("F#4",1)],
        [("B4",.5),("B4",.5),("D5",.5),("E5",.5),("F5",.5),("E5",.5),("D5",1)],
        [("C#5",.5),("A4",.5),("F#4",1),("G4",.5),("A4",.5),("B4",1)],
        [("D5",.5),("F5",.5),("E5",1),("C#5",.5),("B4",.5),("A4",1)],
        [("G4",.5),("B4",.5),("D5",1),("C#5",.5),("A4",.5),("F#4",1)],
        [("B4",.5),("A4",.5),("F#4",.5),("E4",.5),("G4",.5),("A4",.5),("C#5",1)],
        [("B4",.5),("A4",.5),("F#4",1),("C#5",.5),("D5",.5),("B4",1)],
    ]
    bridge: List[Pattern] = [
        [("F#4",1),("A4",.5),("B4",.5),("D5",1),("C#5",1)],
        [("B4",.5),("A4",.5),("F#4",1),("E4",1),(None,1)],
        [("G4",.5),("A4",.5),("B4",1),("D5",.5),("C#5",.5),("A4",1)],
        [("F#4",1),("C#5",.5),("D5",.5),("E5",1),("D5",1)],
        [("B4",.5),("D5",.5),("F5",1),("E5",.5),("D5",.5),("B4",1)],
        [("A4",.5),("B4",.5),("C#5",1),("D5",.5),("C#5",.5),("A4",1)],
        [("G4",1),("B4",.5),("A4",.5),("F#4",1),("E4",1)],
        [("F#4",.5),("A4",.5),("C#5",1),("B4",.5),("A4",.5),("B4",1)],
    ]
    progression = [
        ("B1",["B2","F#3","A3","E4"]),
        ("G1",["G2","D3","F#3","B3"]),
        ("E1",["E2","B2","D3","F#3"]),
        ("F#1",["F#2","C#3","E3","G3"]),
        ("A1",["A2","E3","F#3","B3"]),
        ("D2",["D3","A3","C#4","F#4"]),
        ("E1",["E2","B2","D3","G3"]),
        ("F#1",["F#2","C#3","E3","A#3"]),
    ]
    counter: List[Pattern] = [
        [("B3",2),("F#4",2)],[("G3",2),("D4",2)],
        [("E3",2),("B3",2)],[("F#3",2),("C#4",2)],
        [("A3",2),("E4",2)],[("D3",2),("A3",2)],
        [("E3",2),("B3",2)],[("F#3",2),("C#4",2)],
    ]

    def clav_bar(bar: int, idx: int, full: bool):
        start = bar * 4
        _, tones = progression[idx]
        sequence = [tones[0],tones[1],tones[0],tones[2],tones[1],tones[3],tones[0],tones[1]]
        for step, pitch in enumerate(sequence):
            if not full and step == 3:
                continue
            add(events,"Comp Clav",start+step*.5,pitch,.28,72 if step%4==0 else 56)

    def bass_bar(bar: int, idx: int, dense: bool):
        start = bar * 4
        root, tones = progression[idx]
        if dense:
            sequence = [(0,root),(1.5,tones[1]),(2,nn(root)+12),(3.5,tones[2])]
            duration = .72
        else:
            sequence = [(0,root),(2,nn(root)+12)]
            duration = 1.55
        for number,(offset,pitch) in enumerate(sequence):
            add(events,"Fretless Bass",start+offset,pitch,duration,90 if number==0 else 75)

    def guitar_bar(bar: int, idx: int, enabled: bool):
        if not enabled:
            return
        start = bar * 4
        _, tones = progression[idx]
        for offset in (0,2.5):
            chord(events,"Power Guitar",start+offset,[nn(tones[0])-12,nn(tones[1])-12],.30,66)

    def drum_bar(bar: int, energy: bool):
        start = bar * 4
        kicks = [0,1.5,2.5] if not energy else [0,.75,1.5,2.5,3.5]
        for offset in kicks:
            add(events,"POWER Drums",start+offset,36,.08,96 if offset in (0,2.5) else 78)
        for offset in (1,3):
            add(events,"POWER Drums",start+offset,38,.10,109)
        for step in range(8):
            add(events,"POWER Drums",start+step*.5,42,.055,47 if step%2 else 60)
        if energy and bar%2==1:
            add(events,"POWER Drums",start+2.75,40,.06,68)
        if bar%4==3:
            for step,drum in enumerate([45,47,48,50]):
                add(events,"POWER Drums",start+3+step*.25,drum,.07,82+step*5)
        if bar in (0,8,16,24):
            add(events,"POWER Drums",start,49,.18,104)

    for bar in range(8):
        idx = bar
        start = bar*4
        pattern(events,"Baritone Sax",start,hook[idx],108)
        clav_bar(bar,idx,False)
        bass_bar(bar,idx,False)
        guitar_bar(bar,idx,bar>=4)
        drum_bar(bar,False)
        if idx in (0,4):
            add(events,"Timpani",start,nn(progression[idx][0])+12,.55,80)

    for bar in range(8,16):
        idx = bar-8
        start = bar*4
        pattern(events,"Romantic Trumpet",start,hook[idx],110)
        pattern(events,"Baritone Sax",start,counter[idx],58)
        clav_bar(bar,idx,True)
        bass_bar(bar,idx,True)
        guitar_bar(bar,idx,True)
        drum_bar(bar,True)
        chord(events,"Strings",start,progression[idx][1],3.75,34)
        if idx in (0,4):
            chord(events,"Orchestra Hit",start,[progression[idx][1][0],progression[idx][1][2]],.18,86)

    for bar in range(16,24):
        idx = bar-16
        start = bar*4
        lead = "Baritone Sax" if idx<4 else "Romantic Trumpet"
        answer = "Romantic Trumpet" if idx<4 else "Baritone Sax"
        pattern(events,lead,start,bridge[idx],112)
        if idx%2==1:
            pattern(events,answer,start,counter[idx],55)
        clav_bar(bar,idx,True)
        bass_bar(bar,idx,True)
        guitar_bar(bar,idx,True)
        drum_bar(bar,True)
        chord(events,"Strings",start,progression[idx][1],3.75,46)
        if idx in (3,7):
            chord(events,"Orchestra Hit",start+3.5,[progression[idx][1][0],progression[idx][1][2],progression[idx][1][3]],.28,108)

    for bar in range(24,32):
        idx = bar-24
        start = bar*4
        pattern(events,"Romantic Trumpet",start,hook[idx],114)
        pattern(events,"Baritone Sax",start,hook[idx],72,-12)
        clav_bar(bar,idx,True)
        bass_bar(bar,idx,True)
        guitar_bar(bar,idx,True)
        drum_bar(bar,True)
        chord(events,"Strings",start,progression[idx][1],3.75,52)
        if idx%2==0:
            chord(events,"Choir Aahs",start,progression[idx][1],3.75,30)
        add(events,"25% Pulse",start,nn(progression[idx][1][0])+12,1.6,46)
        add(events,"25% Pulse",start+2,nn(progression[idx][1][1])+12,1.6,42)
        if idx in (0,4):
            add(events,"Timpani",start,nn(progression[idx][0])+12,.62,92)
        if idx in (3,7):
            chord(events,"Orchestra Hit",start+3.5,[progression[idx][1][0],progression[idx][1][2],progression[idx][1][3]],.34,118)

    return events


def render_sf2(path: Path, bank: int, program: int,
               events: Sequence[Event], gain: float) -> np.ndarray:
    total = int((BARS*BAR+TAIL)*SR)
    synth = fluidsynth.Synth(gain=gain,samplerate=SR)
    sfid = synth.sfload(str(path))
    channel = 9 if bank>=120 else 0
    if synth.program_select(channel,sfid,bank,program)!=0:
        raise RuntimeError(f"Preset selection failed: {path} bank={bank} program={program}")
    synth.set_reverb(roomsize=.13,damping=.72,width=.68,level=.08)
    synth.set_chorus(nr=2,level=.10,speed=.22,depth=1.7,type=0)
    timeline=[]
    for start,duration,note,velocity in events:
        timeline.append((int(start*BEAT*SR),1,note,velocity))
        timeline.append((int((start+duration)*BEAT*SR),0,note,0))
    timeline.sort(key=lambda item:(item[0],item[1]))
    chunks=[]
    cursor=0
    for frame,on,note,velocity in timeline:
        if frame>cursor:
            chunks.append(synth.get_samples(frame-cursor)); cursor=frame
        if on: synth.noteon(channel,note,velocity)
        else: synth.noteoff(channel,note)
    if cursor<total: chunks.append(synth.get_samples(total-cursor))
    synth.delete()
    raw=np.concatenate(chunks).astype(np.float64)/32768.0
    return raw.reshape(-1,2)


def render_pulse(events: Sequence[Event]) -> np.ndarray:
    total=int((BARS*BAR+TAIL)*SR)
    audio=np.zeros((total,2),dtype=np.float64)
    for start,duration,note,velocity in events:
        seconds=duration*BEAT
        count=int((seconds+.04)*SR)
        time=np.arange(count)/SR
        frequency=440.0*2.0**((note-69)/12.0)
        signal=np.where((frequency*time)%1.0<.25,1.0,-1.0)
        envelope=np.minimum(1.0,time/.002)*(.30+.70*np.exp(-time/.25))
        gate=min(count,int(seconds*SR))
        if gate<count: envelope[gate:]*=np.linspace(1.0,0.0,count-gate)
        signal=np.round(signal*12.0)/12.0
        signal*=envelope*(velocity/127.0)*.075
        begin=int(start*BEAT*SR); end=min(total,begin+count)
        audio[begin:end,0]+=signal[:end-begin]*.70
        audio[begin:end,1]+=signal[:end-begin]*.74
    return audio


def write_wav(path: Path, audio: np.ndarray) -> None:
    peak=float(np.max(np.abs(audio)))
    if peak>0: audio=audio/peak*.94
    pcm=(np.clip(audio,-1,1)*32767).astype(np.int16)
    with wave.open(str(path),"wb") as handle:
        handle.setnchannels(2); handle.setsampwidth(2); handle.setframerate(SR)
        handle.writeframes(pcm.tobytes())


def export(wav_path: Path, mp3_path: Path, ogg_path: Path|None=None) -> None:
    effect="highpass=f=27,lowpass=f=11500,acompressor=threshold=-17dB:ratio=2.2:attack=8:release=105,alimiter=limit=0.96"
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(wav_path),"-af",effect,"-codec:a","libmp3lame","-q:a","2",str(mp3_path)],check=True)
    if ogg_path:
        subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(wav_path),"-af",effect,"-codec:a","libvorbis","-q:a","6",str(ogg_path)],check=True)


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--undertale",required=True,type=Path)
    parser.add_argument("--sgm1",required=True,type=Path)
    parser.add_argument("--sgm2",required=True,type=Path)
    parser.add_argument("--sgm3",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=True)

    romantic=find_named_preset(args.undertale,[["romantic","tp"],["romantic","trumpet"]])
    orchestra=find_named_preset(args.undertale,[["orchestra","hit"],["orchestrahit"]])
    exact={
        "Comp Clav":(args.sgm3,8,7,"SGM V2.01 Comp Clav"),
        "Baritone Sax":(args.sgm2,0,67,"SGM V2.01 Baritone Sax"),
        "Romantic Trumpet":(args.undertale,romantic[1],romantic[2],romantic[0]),
        "Fretless Bass":(args.sgm2,0,35,"SGM V2.01 Fretless Bass"),
        "Power Guitar":(args.sgm3,16,30,"SGM V2.01 Power Guitar"),
        "Strings":(args.sgm2,0,48,"SGM V2.01 Strings"),
        "Choir Aahs":(args.sgm2,0,52,"SGM V2.01 Choir Aahs"),
        "Orchestra Hit":(args.undertale,orchestra[1],orchestra[2],orchestra[0]),
        "Timpani":(args.sgm2,0,47,"SGM V2.01 Timpani"),
        "POWER Drums":(args.sgm3,128,16,"SGM V2.01 POWER DrumKit"),
    }
    events=build_events()
    gains={
        "Comp Clav":.58,"Baritone Sax":.84,"Romantic Trumpet":.80,
        "Fretless Bass":.64,"Power Guitar":.34,"Strings":.34,
        "Choir Aahs":.22,"Orchestra Hit":.38,"Timpani":.46,"POWER Drums":.72,
    }
    total=int((BARS*BAR+TAIL)*SR)
    full=np.zeros((total,2),dtype=np.float64)
    hook=np.zeros_like(full)
    for label,gain in gains.items():
        path,bank,program,_=exact[label]
        stem=render_sf2(path,bank,program,events[label],.78)
        length=min(total,len(stem)); full[:length]+=stem[:length]*gain
        if label in ("Comp Clav","Baritone Sax","Fretless Bass","POWER Drums"):
            hook[:length]+=stem[:length]*{"Comp Clav":.58,"Baritone Sax":.94,"Fretless Bass":.64,"POWER Drums":.60}[label]
    full+=render_pulse(events["25% Pulse"])
    full=np.tanh(full*1.08); hook=np.tanh(hook*1.05)
    hook_frames=int(8*BAR*SR)
    full_wav=args.out/"KRIS_BAD_MEMORY_EXACT_FULL.wav"
    hook_wav=args.out/"KRIS_BAD_MEMORY_EXACT_HOOK.wav"
    write_wav(full_wav,full)
    write_wav(hook_wav,hook[:hook_frames+int(.6*SR)])
    export(full_wav,args.out/"KRIS_BAD_MEMORY_EXACT_FULL.mp3",args.out/"KRIS_BAD_MEMORY_EXACT_GAME.ogg")
    export(hook_wav,args.out/"KRIS_BAD_MEMORY_EXACT_HOOK.mp3")
    lines=["Exact instrument sources:"]
    for label,(path,bank,program,name) in exact.items():
        lines.append(f"{label}: {name} | {path.name} | bank={bank} preset={program}")
    lines.append("25% Pulse: generated duty-cycle waveform matching the Magical 8bit instrument type")
    lines.append(f"Tempo: {BPM} BPM")
    (args.out/"KRIS_BAD_MEMORY_EXACT_INSTRUMENTS.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines))


if __name__=="__main__":
    main()
