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
from mido import MidiFile
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

# file: midi file to be processed.
# granularity: level of detail of the rhythm pattern (ex: 16th notes). always a factor of 2
# rhythm_only: if False, the chords which are played will be determined.
# returns: 2 arrays of length granularity * NUM_MEASURES. TODO: may need condense to array of tuples.
#   1) Note played for each granularity note in the midi
#   2) Rhythm pattern for each granularity note in the midi
# TODO: if file was not fully read, consider indicating so.
def ProcessMidi(file, granularity=16, rhythm_only=False):
    midi = MidiFile(file, clip=True)
    midi_notes = ['--' for note in range(granularity*NUM_MEASURES)]
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

                        chord = GetChord(keys_on)
                        for t in range(time_idx, time_idx+delta_time):
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
def ConcatMidiDatas(lead_notes, lead_rhythm, drum_1, drum_2, granularity=16):
    image = np.zeros((1, 4, len(lead_notes), 1)) # describe with variables
    for i in range(len(lead_notes)):
        # encode note
        notenum = 0
        for j in range(len(TREBEL_NOTES)):
            if TREBEL_NOTES[j]==lead_notes[i][0]:
                notenum=j
                break

        if lead_notes[i][1]=='i':
            notenum+=12

        image[0][0][i]=notenum
        image[0][1][i]=lead_rhythm[i]
        image[0][2][i]=drum_1[i]
        image[0][3][i]=drum_2[i]
#endfor
    return image

# TODO: add a print function to show the notes at a certain granularity
