from pathlib import Path
import argparse, math, wave, subprocess
import numpy as np
import fluidsynth
from mido import MidiFile, MidiTrack, Message, MetaMessage, bpm2tempo

SR = 44100
BPM = 142
BEAT = 60.0 / BPM
BAR = 4 * BEAT
BARS = 32
TPB = 480

PITCH = {
    "C":0,"C#":1,"Db":1,"D":2,"D#":3,"Eb":3,"E":4,"F":5,
    "F#":6,"Gb":6,"G":7,"G#":8,"Ab":8,"A":9,"A#":10,"Bb":10,"B":11
}
def mn(note):
    if isinstance(note, int):
        return note
    if len(note) >= 3 and note[1] in "#b":
        p, o = note[:2], int(note[2:])
    else:
        p, o = note[0], int(note[1:])
    return 12*(o+1)+PITCH[p]

def hz(note):
    return 440.0 * 2 ** ((mn(note)-69)/12)

TRACKS = {
    "Piano 1": [],
    "Strings": [],
    "Choir Aahs": [],
    "French Horns": [],
    "Trumpet": [],
    "Timpani": [],
    "OrchestraHit": [],
    "Standard Drums": [],
    "Orchestra Drums": [],
    "25% Pulse": [],
    "12.5% Pulse": [],
    "Triangle Bass": [],
    "White Noise": [],
}

def add_note(track, start, note, dur, vel=100):
    TRACKS[track].append((float(start), float(dur), mn(note), int(vel)))

def add_pattern(track, start, pattern, vel=100, shift=0):
    t = float(start)
    for n,d in pattern:
        if n is not None:
            add_note(track, t, mn(n)+shift, d*0.94, vel)
        t += d

def add_chord(track, start, notes, dur, vel=70):
    for n in notes:
        add_note(track,start,n,dur,vel)

PROG = [
    ("G2",  ["G3","D4","F4","Bb4"]),
    ("Eb2", ["Eb3","Bb3","D4","G4"]),
    ("F2",  ["F3","C4","Eb4","A4"]),
    ("D2",  ["D3","A3","C4","F#4"]),
    ("G2",  ["G3","D4","F4","Bb4"]),
    ("Bb1", ["Bb2","F3","A3","D4"]),
    ("C2",  ["C3","G3","Bb3","Eb4"]),
    ("D2",  ["D3","A3","C4","F#4"]),
]

THEME = [
    [(None,.5),("D5",.5),("G5",1),("F5",.5),("D5",.5),("Bb4",1)],
    [("D5",.5),("Eb5",.5),("G5",1),("A5",.5),("G5",.5),("D5",1)],
    [(None,.5),("C5",.5),("F5",1),("Eb5",.5),("C5",.5),("A4",1)],
    [("C5",.5),("D5",.5),("F5",1),("Eb5",1),("D5",1)],
    [("D5",.5),("G5",.5),("Bb5",1),("A5",.5),("G5",.5),("F5",1)],
    [("D5",.5),("F5",.5),("A5",1),("G5",.5),("F5",.5),("D5",1)],
    [("C5",.5),("D5",.5),("G5",1),("F5",.5),("Eb5",.5),("D5",1)],
    [("C5",1),("D5",1),("G4",2)],
]

COUNTER = [
    [("Bb3",1),("D4",1),("C4",1),("G3",1)],
    [("G3",1),("Bb3",1),("A3",1),("F#3",1)],
    [("A3",1),("C4",1),("Bb3",1),("F3",1)],
    [("F#3",1),("A3",1),("C4",1),("D4",1)],
    [("G3",1),("Bb3",1),("D4",1),("C4",1)],
    [("F3",1),("A3",1),("D4",1),("Bb3",1)],
    [("Eb3",1),("G3",1),("Bb3",1),("C4",1)],
    [("F#3",1),("A3",1),("C4",1),("D4",1)],
]

B_THEME = [
    [("Bb4",1),("D5",.5),("Eb5",.5),("G5",1),("F5",1)],
    [("A4",1),("C5",.5),("D5",.5),("F5",1),("Eb5",1)],
    [("G4",.5),("Bb4",.5),("D5",1),("C5",1),("Bb4",1)],
    [("A4",1),("C5",1),("D5",.5),("F#5",.5),("G5",1)],
    [("D5",1),("G5",1),("Bb5",.5),("A5",.5),("G5",1)],
    [("F5",1),("D5",1),("C5",.5),("D5",.5),("F5",1)],
    [("Eb5",.5),("F5",.5),("G5",1),("Bb5",1),("A5",1)],
    [("F#5",1),("A5",1),("G5",2)],
]

def add_bass_bar(bar, dense=False):
    s = bar*4
    root,ch = PROG[bar%8]
    if dense:
        seq=[root,ch[0],root,ch[1],root,ch[2],root,ch[3]]
        for i,n in enumerate(seq):
            add_note("Triangle Bass",s+i*.5,n,.40,88 if i%2==0 else 72)
    else:
        add_note("Triangle Bass",s,root,1.7,92)
        add_note("Triangle Bass",s+2,ch[0],1.7,78)

def add_piano_ostinato(bar, strong=False):
    s=bar*4
    _,ch=PROG[bar%8]
    seq=[ch[0],ch[2],ch[1],ch[3],ch[0],ch[2],ch[1],ch[3]]
    for i,n in enumerate(seq):
        add_note("Piano 1",s+i*.5,n,.34,72 if strong else 58)

def add_drums(bar, intensity=.8, fill=False):
    s=bar*4
    kicks = [0, 1.5, 2.5] if bar%2==0 else [0, 1.0, 2.75]
    snares = [1,3]
    for k in kicks:
        add_note("Standard Drums",s+k,36,.08,int(104*intensity))
    for sn in snares:
        add_note("Standard Drums",s+sn,38,.10,int(116*intensity))
    for i in range(8):
        add_note("Standard Drums",s+i*.5,42,.06,int((56 if i%2 else 68)*intensity))
    if fill:
        for i,n in enumerate([45,47,48,50]):
            add_note("Standard Drums",s+3+i*.25,n,.08,88+i*7)
        add_note("Orchestra Drums",s+3.75,49,.18,108)

for bar in range(8):
    s=bar*4; _,ch=PROG[bar]
    add_bass_bar(bar,False); add_piano_ostinato(bar,False)
    add_pattern("Piano 1",s,THEME[bar],94)
    add_pattern("25% Pulse",s,THEME[bar],46,shift=-12)
    add_chord("Strings",s,ch,3.72,38)
    if bar>=2: add_drums(bar,.68,fill=(bar==7))
    if bar in (0,4): add_chord("OrchestraHit",s,[ch[0],ch[2],ch[3]],.18,90)

for bar in range(8,16):
    s=bar*4; idx=bar-8; _,ch=PROG[idx]
    add_bass_bar(bar,False); add_piano_ostinato(bar,True)
    add_pattern("French Horns",s,THEME[idx],96)
    add_pattern("Piano 1",s,COUNTER[idx],62)
    add_chord("Strings",s,ch,3.76,54)
    if bar%2==0: add_chord("Choir Aahs",s,ch,3.7,35)
    add_drums(bar,.84,fill=(bar%4==3))
    if bar in (8,12): add_note("Timpani",s,mn(PROG[idx][0])+12,.45,100)

for bar in range(16,24):
    s=bar*4; idx=bar-16; _,ch=PROG[(idx+2)%8]
    add_bass_bar(bar,False)
    add_pattern("Trumpet",s,B_THEME[idx],94)
    add_pattern("12.5% Pulse",s,COUNTER[(idx+2)%8],40,shift=12)
    add_chord("Strings",s,ch,3.76,58); add_chord("Choir Aahs",s,ch,3.76,40)
    add_drums(bar,.80,fill=(bar==23))
    for off in (0,2): add_note("Timpani",s+off,mn(PROG[(idx+2)%8][0])+12,.30,88)

for bar in range(24,32):
    s=bar*4; idx=bar-24; _,ch=PROG[idx]
    add_bass_bar(bar,True); add_piano_ostinato(bar,True)
    add_pattern("French Horns",s,THEME[idx],106)
    add_pattern("25% Pulse",s,THEME[idx],62,shift=12 if bar>=28 else 0)
    add_pattern("Piano 1",s,COUNTER[idx],66)
    add_chord("Strings",s,ch,3.76,62); add_chord("Choir Aahs",s,ch,3.76,46)
    add_drums(bar,.96,fill=(bar%4==3))
    if bar%2==0: add_chord("OrchestraHit",s,[ch[0],ch[2],ch[3]],.18,104)
    for off in (0,2): add_note("Timpani",s+off,mn(PROG[idx][0])+12,.32,96)

end=32*4-.5
add_chord("Piano 1",end,["G3","D4","G4","Bb4"],.45,120)
add_chord("OrchestraHit",end,["G3","D4","F4","Bb4"],.35,124)
add_note("Standard Drums",end,49,.25,127); add_note("Timpani",end,"G3",.45,127)
for b in [8*4-.25,16*4-.25,24*4-.25,32*4-.75]: TRACKS["White Noise"].append((b,.20,60,92))

SAMPLE_SPECS = {
    "Piano 1": ("part1",0,0), "Strings": ("part2",0,48),
    "Choir Aahs": ("part2",0,52), "French Horns": ("part2",0,60),
    "Trumpet": ("part2",0,56), "Timpani": ("part2",0,47),
    "OrchestraHit": ("part2",0,55), "Standard Drums": ("part3",128,0),
    "Orchestra Drums": ("part3",128,48),
}

def render_sf2(sf_path, bank, preset, events, gain=.75, tail=1.4):
    total=int((BARS*BAR+tail)*SR)
    synth=fluidsynth.Synth(gain=gain,samplerate=SR)
    sfid=synth.sfload(str(sf_path)); ch=9 if bank==128 else 0
    synth.program_select(ch,sfid,bank,preset)
    synth.set_reverb(roomsize=.16,damping=.68,width=.70,level=.10)
    synth.set_chorus(nr=2,level=.15,speed=.25,depth=2.0,type=0)
    timeline=[]
    for st,dur,n,v in events:
        timeline += [(int(st*BEAT*SR),1,n,v),(int((st+dur)*BEAT*SR),0,n,0)]
    timeline.sort(key=lambda x:(x[0],x[1]))
    chunks=[]; cur=0
    for frame,on,n,v in timeline:
        if frame>cur: chunks.append(synth.get_samples(frame-cur)); cur=frame
        if on: synth.noteon(ch,n,v)
        else: synth.noteoff(ch,n)
    if total>cur: chunks.append(synth.get_samples(total-cur))
    synth.delete()
    if not chunks: return np.zeros((total,2))
    return (np.concatenate(chunks).astype(np.float64)/32768.0).reshape(-1,2)

def add_custom(buffer,sig,start,pan=0):
    a=int(start*BEAT*SR); b=min(len(buffer),a+len(sig))
    if b<=a: return
    ang=(pan+1)*math.pi/4
    buffer[a:b,0]+=sig[:b-a]*math.cos(ang); buffer[a:b,1]+=sig[:b-a]*math.sin(ang)

def pulse(note,dur_b,vel,duty):
    dur=dur_b*BEAT; n=int((dur+.04)*SR); t=np.arange(n)/SR; f=hz(note)
    sig=np.where((f*t)%1<duty,1.0,-1.0); sig=np.round(sig*10)/10
    env=np.minimum(1,t/.002)*(.24+.76*np.exp(-t/.28)); g=min(n,int(dur*SR))
    if g<n: env[g:]*=np.linspace(1,0,n-g)
    return sig*env*(vel/127)*.095

def triangle(note,dur_b,vel):
    dur=dur_b*BEAT; n=int((dur+.04)*SR); t=np.arange(n)/SR; f=hz(note)
    ph=(f*t)%1; sig=4*np.abs(ph-.5)-1; sig=np.round(sig*14)/14
    env=np.minimum(1,t/.003)*(.25+.75*np.exp(-t/.38)); g=min(n,int(dur*SR))
    if g<n: env[g:]*=np.linspace(1,0,n-g)
    return sig*env*(vel/127)*.15

def noise(dur_b,vel):
    dur=dur_b*BEAT; n=int(dur*SR); t=np.arange(n)/SR
    rng=np.random.default_rng(142); sig=rng.integers(-8,9,size=n)/8
    return sig*np.exp(-t/max(.01,dur*.18))*(vel/127)*.06

def export_midi(path):
    programs={"Piano 1":0,"Strings":48,"Choir Aahs":52,"French Horns":60,
              "Trumpet":56,"Timpani":47,"OrchestraHit":55,"25% Pulse":80,
              "12.5% Pulse":80,"Triangle Bass":38,"Standard Drums":0,"Orchestra Drums":48}
    channels={name:i for i,name in enumerate(programs)}
    channels["Standard Drums"]=9; channels["Orchestra Drums"]=9
    mid=MidiFile(ticks_per_beat=TPB); tt=MidiTrack()
    tt.append(MetaMessage("track_name",name="Tempo",time=0))
    tt.append(MetaMessage("set_tempo",tempo=bpm2tempo(BPM),time=0))
    tt.append(MetaMessage("time_signature",numerator=4,denominator=4,time=0)); mid.tracks.append(tt)
    for name in programs:
        tr=MidiTrack(); tr.append(MetaMessage("track_name",name=name,time=0)); ch=channels[name]
        tr.append(Message("program_change",channel=ch,program=programs[name],time=0)); line=[]
        for st,dur,n,v in TRACKS[name]: line += [(round(st*TPB),1,n,v),(round((st+dur)*TPB),0,n,0)]
        line.sort(key=lambda x:(x[0],x[1])); last=0
        for tick,on,n,v in line:
            tr.append(Message("note_on" if on else "note_off",channel=ch,note=n,velocity=v if on else 0,time=tick-last)); last=tick
        tr.append(MetaMessage("end_of_track",time=1)); mid.tracks.append(tr)
    mid.save(path)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--part1",required=True); ap.add_argument("--part2",required=True)
    ap.add_argument("--part3",required=True); ap.add_argument("--out",required=True); args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    parts={"part1":Path(args.part1),"part2":Path(args.part2),"part3":Path(args.part3)}
    total=int((BARS*BAR+1.4)*SR); mix=np.zeros((total,2),dtype=np.float64)
    gains={"Piano 1":.78,"Strings":.62,"Choir Aahs":.45,"French Horns":.72,"Trumpet":.66,
           "Timpani":.70,"OrchestraHit":.72,"Standard Drums":.80,"Orchestra Drums":.55}
    for name,(part,bank,preset) in SAMPLE_SPECS.items():
        stem=render_sf2(parts[part],bank,preset,TRACKS[name],.72)
        if "Drums" in name: stem=np.tanh(stem*1.35)
        if name=="Trumpet": stem=np.tanh(stem*1.10)
        n=min(len(mix),len(stem)); mix[:n]+=stem[:n]*gains[name]
    custom=np.zeros_like(mix)
    for st,dur,n,v in TRACKS["25% Pulse"]: add_custom(custom,pulse(n,dur,v,.25),st,.10)
    for st,dur,n,v in TRACKS["12.5% Pulse"]: add_custom(custom,pulse(n,dur,v,.125),st,-.12)
    for st,dur,n,v in TRACKS["Triangle Bass"]: add_custom(custom,triangle(n,dur,v),st,-.03)
    for st,dur,n,v in TRACKS["White Noise"]: add_custom(custom,noise(dur,v),st,0)
    audio=mix*.92+custom; dry=audio.copy()
    for delay,amt in [(0.055,.018),(0.11,.012)]:
        d=int(delay*SR); audio[d:]+=dry[:-d]*amt
    audio=np.tanh(audio*1.12); peak=np.max(np.abs(audio))
    if peak: audio=audio/peak*.95
    wav=out/"KRIS_A_NAME_WITHOUT_A_FACE.wav"; mp3=out/"KRIS_A_NAME_WITHOUT_A_FACE.mp3"
    ogg=out/"KRIS_A_NAME_WITHOUT_A_FACE_GAME.ogg"; midi=out/"KRIS_A_NAME_WITHOUT_A_FACE.mid"
    with wave.open(str(wav),"wb") as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SR); wf.writeframes((audio*32767).astype(np.int16).tobytes())
    af="highpass=f=28,acompressor=threshold=-17dB:ratio=2:attack=8:release=100,alimiter=limit=0.96"
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(wav),"-af",af,"-codec:a","libmp3lame","-q:a","2",str(mp3)],check=True)
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(wav),"-af",af,"-codec:a","libvorbis","-q:a","6",str(ogg)],check=True)
    export_midi(midi); print(mp3); print(ogg); print(midi)

if __name__=="__main__": main()
