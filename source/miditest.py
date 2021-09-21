# miditest.py
# Purpose: tests midi handling

import os
from os import listdir
from os.path import isfile, join
import mido
from mido import MidiFile

TRAINING_DIR = "../data/train"  # reminder: train is split into leads and drums

lead_files = [join(f'{TRAINING_DIR}/leads/',f) for f in listdir(f'{TRAINING_DIR}/leads') if isfile(join(f'{TRAINING_DIR}/leads/', f) )]
drum_files = [join(f'{TRAINING_DIR}/drums/',f) for f in listdir(f'{TRAINING_DIR}/drums') if isfile(join(f'{TRAINING_DIR}/drums/', f) )]

print(lead_files)
print(drum_files)

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
