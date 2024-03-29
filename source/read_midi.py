"""
read_midi.py
Author: Michael Probst
Purpose: This file contains functions that read a midi file and uses it to fill
an array spanning 4 measures of beats, of potentially varying note granularity.
Output array is 2 dimesional where second dimension is an array containing
[lead_note_id, midi_rhythm, drum_rhythm].
"""

import os
from os import listdir
from os.path import isfile, join
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
from math import floor
from math import ceil
import numpy as np

# Upper case for standard note, lowercase for sharp
TREBEL_NOTES = ['C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g', 'A', 'a', 'B']
# Upper case for standard note, lowercase for flat
BASS_NOTES = ['A', 'a', 'B', 'C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g']

MIDI_LENGTH = 128
NUM_MEASURES = 4        # number of measures to observe from the midi

# note: single midi note
# returns: corresponding standard piano note
def GetClefNote(note):
    clef = TREBEL_NOTES
    return clef[note % 12]

# notes: an array of midi notes which are currently played
# returns: chord or note describing the input array
def GetChord(notes):
    if len(notes) == 0:
        return "--"

    root = -1
    isMajor = True
    # not enough for a chord, return highest note
    if len(notes) <= 2:
        return f'{GetClefNote(notes[len(notes)-1])}-'

    # if split between voices, drop outliers an octave
    # find if there is a group of common notes
    # if we have 3 notes within 7 hs, we can use those to id the chord
    for i in range(len(notes)-2):
        if not root == -1 or i+2 >= len(notes):
            break

        chord = notes[i : (i+3)]

        # check for split voices
        rhct = 0    # right hand count
        lhct = 0    # left hand count
        for c in chord:
            if c < 60:  # middle c is note 60
                rhct+=1
            else:
                lhct+=1

        # we have split voices, normalize the chord
        if rhct > 0 and lhct > 0:
            if rhct < lhct:
                # get how many octaves lower, then increase it by that much
                chord[0]+=floor((chord[1]-chord[0])/8)*8
            else:
                # get how many octaves higher, then decrease it by that much
                chord[2]-=floor((chord[2]-chord[1])/8)*8

        chord.sort()
        diff = chord[2] - chord[0]
        split_a = chord[1] - chord[0]
        split_b = chord[2] - chord[1]

        # check for inversions, then normalize chord
        if diff > 7:
            if split_b == 5:
                #print("first inversion")
                chord[2]-=12
            elif split_a == 5:
                #print("second inversion")
                chord[0]+=12
            chord.sort()

        isMajor = chord[1]-chord[0] == 4

        # at this point, the chord should always have a diff of 7hs
        if chord[2]-chord[0] == 7:
            root = chord[0]
    # if the same note appears in the natural form in the scale of the chord, then it is a flat,
    #   otherwise it is a sharp. for now, we know if we see C# major, then this is really Db major.
    # endfor
    if root == -1:
        return f'{GetClefNote(notes[len(notes)-1])}-'

    return f'{GetClefNote(root)}{"j" if isMajor else "i"}'

# encodes a chord into a distinct integer
# chordstring: first char is chord letter, second is minor or major indicator
def EncodeChord(chordstring):
    # encode note
    notenum = 0
    for j in range(len(TREBEL_NOTES)):
        if TREBEL_NOTES[j]==chordstring[0]:
            notenum=j
            break

    if chordstring[1]=='i':
        notenum+=12

    return notenum

# file: midi file to be processed.
# granularity: level of detail of the rhythm pattern (ex: 16th notes). always a factor of 2
# rhythm_only: if False, the chords which are played will be determined.
# returns: 2D array of length granularity * NUM_MEASURES. TODO: may need condense to array of tuples.
#   1) Note played for each granularity note in the midi
#   2) Rhythm pattern for each granularity note in the midi
# TODO: if file was not fully read, consider indicating so.
def ProcessMidi(file, granularity=16, rhythm_only=False):
    midi = MidiFile(file, clip=True)
    print(f'FILE: {file}\n{midi}')
    midi_notes = [0 for note in range(granularity*NUM_MEASURES)]
    midi_rhythm = [0 for note in range(granularity*NUM_MEASURES)]
    notes_on = [0 for note in range(MIDI_LENGTH)]

    midi_clk = 24   # TODO: get this from the meta message. standard is 24
    pat_clk = int(midi_clk * (16 / granularity))   # converts midi time to my rhythm time
    for track in midi.tracks:
        time_idx = 0
        rhythm = 0
        for msg in track:
            if time_idx >= len(midi_notes):
                break
            if msg.type == 'note_on' or msg.type == 'note_off':
                # if delta time > 0 we have advanced time
                if int(msg.time) >= pat_clk:
                    # THIS WORKS FOR 16th NOTES ONLY
                    delta_time = ceil(msg.time / pat_clk)

                    if not rhythm_only:
                        # record what notes were on for the time chunk
                        keys_on = []
                        for j in range(MIDI_LENGTH):
                            if notes_on[j] == 1:
                                keys_on.append(j)

                        chord = EncodeChord(GetChord(keys_on))
                        for t in range(time_idx, min(time_idx+delta_time, len(midi_notes))):
                            midi_notes[t] = chord
                    #endif

                    midi_rhythm[time_idx] = rhythm
                    time_idx += delta_time
                    rhythm = 0
                #endif

                # update the current note information
                if msg.type == 'note_on':
                    notes_on[msg.note]=1
                    rhythm = 1
                if msg.type == 'note_off':
                    notes_on[msg.note]=0
            #endif
        #endfor
    #endfor
    return midi_notes, midi_rhythm

# Concats processed midi arrays into a 2D image to use as input to a CNN
# lead_notes: array of notes played for a midi file
# lead_rhythm: array of rhythm in which lead_notes is played
# drum_1: array describing accompanying slower drum pattern (ex: kick drum, snare)
# drum_2: array describing accompanying consistent drum pattern (ex: high-hats)
#   consistent meaning most of the notes occur in a very repeatable pattern (ex: all quarter notes)
def ConcatMidiDatas(first, second):
    image = np.zeros((1, 2, len(first), 1)) # describe with variables
    for i in range(len(first)):
        image[0][0][i]=first[i]
        image[0][1][i]=second[i]
        #image[0][2][i]=drum_1[i]
        #image[0][3][i]=drum_2[i]
    #endfor
    return image

# TODO: add a print function to show the notes at a certain granularity

def GetTempo(midi):
    for msg in midi.tracks[0]:
        if msg.type == 'set_tempo':
            return msg.tempo

    return 50000

def GetTimeSig(midi):
    for msg in midi.tracks[0]:
        if msg.type == 'time_signature':
            return msg.numerator, msg.denominator

    return 4, 4

def GetTicksPerBeat(midi):
    for msg in midi.tracks[0]:
        if msg.type == 1:
            return msg.value

    return 96


def CreateMidi(data, tracks, path, tik=96, tempo=50000, ts=(4,4), granularity=16):
    track_len = int(len(data) / tracks)
    notes = [36, 42]
    thresh = [0.40, 0.25]    # be more permissive with high hats
    for i in range(tracks):
        mid = MidiFile(ticks_per_beat=tik)
        meta = MidiTrack()
        meta.append(MetaMessage('set_tempo', tempo=tempo, time=0))
        meta.append(MetaMessage('time_signature', numerator=ts[0], denominator=ts[1], clocks_per_click=24))
        mid.tracks.append(meta)

        track = MidiTrack()
        mid.tracks.append(track)

        note = notes[i]
        note_len = int(24 * (16/granularity))
        for j in range(track_len):
            print(f'note[{j}]={data[j]}')
            if data[j+i*track_len] < thresh[i]:
                track.append(Message('note_off', note=note, velocity=64, time=0))
            else:
                track.append(Message('note_on', note=note, velocity=64, time=0))
            track.append(Message('note_off', note=note, velocity=64, time=note_len))

        #print(mid)
        mid.save(f'{path}gen_{i+1}.mid')
