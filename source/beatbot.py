"""
beatbot.py
Author: Michael Probst
Purpose: With a data set of midi files of lead instruments and their
         accompanying drum midis, this agent learns how to create hip-hop/rap
         beats using a CNN.
"""

import os
from os import listdir
from os.path import isfile, join, isdir

# project files
from cnn import CNN
import read_midi

TRAINING_DIR = "../data/train"  # reminder: train is split into leads and drums
GRANULARITY = 16
NUM_MEASURES = 4

lead_files = [join(f'{TRAINING_DIR}/leads/',f)
              for f in listdir(f'{TRAINING_DIR}/leads')
                if isfile(join(f'{TRAINING_DIR}/leads/', f))
             ]
drum_folders = [join(f'{TRAINING_DIR}/drums/',f)
                for f in listdir(f'{TRAINING_DIR}/drums')
                    if isdir(join(f'{TRAINING_DIR}/drums/', f))
               ]
# Array of tuples containig both drum files.
# Using 2 drum files because in practice, drum patterns are split into several
# instrumensts/tracks
drum_files = []
for folder in drum_folders:
    files = [join(f'{folder}/',f)
             for f in listdir(f'{folder}/')
                if isfile(join(f'{folder}/',f))
            ]
    drum_files.append(files)

images = [] # inputs to the CNN

for i in range(len(lead_files)):
    lead_notes, lead_rhythm = read_midi.ProcessMidi(lead_files[i], GRANULARITY)
    _, drum_1 = read_midi.ProcessMidi(drum_files[i][0], GRANULARITY, True)
    _, drum_2 = read_midi.ProcessMidi(drum_files[i][1], GRANULARITY, True)
    image = read_midi.ConcatMidiDatas(lead_notes, lead_rhythm, drum_1, drum_2, GRANULARITY)
    images.append(image)

net = CNN(4, GRANULARITY*NUM_MEASURES, 300)
out = []
for image in images:
    out = net.forward(image)

print(out)
