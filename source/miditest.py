# miditest.py
# Purpose: tests midi handling

import os
from os import listdir
from os.path import isfile, join
import mido
from mido import MidiFile
from math import floor
from math import ceil
import torch as T
import numpy as np
import cnn

# Upper case for standard note, lowercase for sharp
TREBEL_NOTES = ['C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g', 'A', 'a', 'B']
# Upper case for standard note, lowercase for flat
BASS_NOTES = ['A', 'a', 'B', 'C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g']


NOTES_PER_MEASURE = 4   # number of beats/notes in a measure
NOTE_LENGTH = 4         # indicates the type of note which counts the pulse. always a factor of 2
                        # NOTES_PER_MEASURE / NOTE_LENGTH = time signature.
GRANULARITY = 16         # level of detail of the rhythm pattern (ex: 16th notes). always a factor of 2
PATTERN_LENGTH = 4      # number of measures we are observing at a time.
NUM_OBSERVATIONS = 1    # number of measures summarized when observing (can be <1)
                        # ex: 1 would print out "measure 1 plays a C chord in rhythm pattern 1"
                        # number of summaries = PATTERN_LENGTH / NUM_OBSERVATIONS
MIDI_NOTES = 128

# we will describe small lengths of midi files patterns as is convention in FL Studio
PATTERN_SIZE = int(PATTERN_LENGTH * NOTES_PER_MEASURE * GRANULARITY / NOTE_LENGTH)
# TODO: check midi file for its time signature

TRAINING_DIR = "../data/train"  # reminder: train is split into leads and drums

def GetClefNote(note):
    clef = TREBEL_NOTES
    #if note < 60:    # middle c is note 60
    #    clef = BASS_NOTES

    #print(f'note {note} = {clef[note%12]}')
    # need note relative to next lowest c.
    # 60 is middle c, there are
    return clef[note % 12]

def GetChord(notes):
    #print(f'getting chord for: {notes}')
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
        #print(f'chord: {chord}')

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
                #print(f'lower split')
                chord[0]+=floor((chord[1]-chord[0])/8)*8
            else:
                # get how many octaves higher, then decrease it by that much
                #print(f'higher split')
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
        print(f'adjusted chord: {chord}')
        # at this point, the chord should always have a diff of 7hs
        if chord[2]-chord[0] == 7:
            root = chord[0]
    # need to determine if chord is a flat or sharp. TODO
    # if the same note appears in the natural form in the scale of the chord, then it is a flat,
    #   otherwise it is a sharp. for now, we know if we see C# major, then this is really Db major.
    # endfor
    if root == -1:
        return f'{GetClefNote(notes[len(notes)-1])}-'
    # now we have the root, we can determine if it is major or minor
    return f'{GetClefNote(root)}{"j" if isMajor else "i"}'

def main():
    lead_files = [join(f'{TRAINING_DIR}/leads/',f) for f in listdir(f'{TRAINING_DIR}/leads') if isfile(join(f'{TRAINING_DIR}/leads/', f) )]
    drum_files = [join(f'{TRAINING_DIR}/drums/',f) for f in listdir(f'{TRAINING_DIR}/drums') if isfile(join(f'{TRAINING_DIR}/drums/', f) )]

    # goal: get what notes are on for a given measure of the song
    #       identify the chord played and the rhythm pattern from a set of
    #       predefined patterns. Based on this, the nn will be able to see what
    #       drum rhythm patterns are commonly associated with lead ryhthm patterns

    # patterns are arrays of tuples describing the note played (chord) and if held or just pressed
        # NOTE: pattern arrays are not yet tuples, but that might be good later

    notes_on = [0 for note in range(MIDI_NOTES)]

    print(lead_files)
    print(drum_files)

    for i in range(len(lead_files)):
        lead = MidiFile(lead_files[i], clip=True)
        drum = MidiFile(drum_files[i], clip=True)
        lead_notes = [0 for note in range(PATTERN_SIZE)]
        lead_rhythm = [0 for note in range(PATTERN_SIZE)]
        drum_pattern = [0 for note in range(PATTERN_SIZE)]
        # okay, lets look at the lead file and see if we can figure out what chord is played

        # fill out an array describing the note played and for how long.
        # midi_clk is the number of events sent per 16th note
        midi_clk = 24   # TODO: get this from the meta message
        pat_clk = int(midi_clk * (16 / GRANULARITY))   # converts midi time to my rhythm time
        song_length = 0
        last_note = 0
        for track in lead.tracks:
            time_idx = 0
            for msg in track:
                if msg.type == 'note_on' or msg.type == 'note_off':
                    # if delta time > 0 we have advanced time
                    song_length += msg.time
                    if int(msg.time) >= pat_clk:
                        # THIS WORKS FOR 16th NOTES ONLY
                        delta_time = ceil(msg.time / pat_clk)

                        # record what notes were on for the time chunk
                        print(f'idx: {time_idx} at time: {song_length} next is: {msg.time/pat_clk} = {delta_time}')

                        keys_on = []
                        for j in range(MIDI_NOTES):
                            if notes_on[j] == 1:
                                keys_on.append(j)

                        chord = GetChord(keys_on)
                        print(f'chord: {chord}')

                        for t in range(time_idx, time_idx+delta_time):
                            lead_notes[t] = chord

                        lead_rhythm[time_idx] = rhythm
                        time_idx += delta_time

                        rhythm = 0

                    # update the current note information
                    if msg.type == 'note_on':
                        notes_on[msg.note]=1
                        rhythm = 1
                    if msg.type == 'note_off':
                        notes_on[msg.note]=0

        song_length = int(song_length/midi_clk)
        print(f'song_length: {song_length}, pat_len: {PATTERN_SIZE}')

        print(f'lead 1/{GRANULARITY} notes')
        #print(lead_notes)
        image = np.zeros((1, 3, 64, 1)) # describe with variables
        for measure in range(PATTERN_LENGTH):
            print(f'measure: {measure+1}')

            for note in range(GRANULARITY):
                idx = measure*GRANULARITY + note
                print(f'{note+1}/{GRANULARITY} - NOTE: {lead_notes[idx]} RYTHM: {lead_rhythm[idx]} idx: {idx}')
                notenum = 0
                for i in range(len(TREBEL_NOTES)):
                    if TREBEL_NOTES[i]==lead_notes[idx][0]:
                        notenum=i
                        break

                if lead_notes[idx][1]=='i':
                    notenum+=12

                image[0][0][idx]=notenum
                image[0][1][idx]=lead_rhythm[idx]
                image[0][2][idx]=0

        #tensor = T.from_numpy(image)
        #print(image)
        net = cnn.CNN(12)
        out = net.forward(image)
        print(out)

    for file in lead_files:
        leadmid = MidiFile(file, clip=True)
        #print(leadmid)

        # this example is very clean with mostly on/off messages


    for file in drum_files:
        drummid = MidiFile(file, clip=True)
        #print(drummid)
        # looking at individual tracks: mid.tracks
        # mid.tracks[0]: system messages such as: time signature and framerate
        # mid.tracks[1] system messages for tempo.
        # mid.tracks[2 to n] are the actual musical information
            # track[2] had a lot of control and program changes
            # tracks[n] seems to have most of the note on/off data

        # my example from FL may have been messy due to the FL keys info

main()
