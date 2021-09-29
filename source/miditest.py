# miditest.py
# Purpose: tests midi handling

import os
from os import listdir
from os.path import isfile, join
import mido
from mido import MidiFile
from math import floor
from math import ceil

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

                    lead_rhythm[time_idx] = rhythm
                    for j in range(MIDI_NOTES):
                        if notes_on[j] == 1:
                            # todo: get the chord played
                            for t in range(time_idx, time_idx+delta_time, pat_clk):
                                lead_notes[t] = j
                    rhythm = 0
                    time_idx += delta_time
                else:
                    # update the current information
                    if msg.type == 'note_on':
                        notes_on[msg.note]=1
                        rhythm = 1
                    if msg.type == 'note_off':
                        notes_on[msg.note]=0

    song_length = int(song_length/midi_clk)
    print(f'song_length: {song_length}, pat_len: {PATTERN_SIZE}')

    print(f'lead 1/{GRANULARITY} notes')
    #print(lead_notes)
    for measure in range(PATTERN_LENGTH):
        print(f'measure: {measure+1}')

        for note in range(GRANULARITY):
            idx = measure*GRANULARITY + note
            print(f'{note+1}/{GRANULARITY} - NOTE: {lead_notes[idx]} RYTHM: {lead_rhythm[idx]} idx: {idx}')




for file in lead_files:
    leadmid = MidiFile(file, clip=True)
    print(leadmid)

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
