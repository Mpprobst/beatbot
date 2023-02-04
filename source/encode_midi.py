import os
from os import listdir
from os.path import isfile, join, isdir
from shutil import copyfile
import mido
import argparse

import read_midi

TRAIN_DIR = "../data/train"  # reminder: train is split into leads and drums
OUT_DIR = "../output/encoded"

lead_files = [join(f'{TRAIN_DIR}/leads/',f)
              for f in listdir(f'{TRAIN_DIR}/leads')
                if isfile(join(f'{TRAIN_DIR}/leads/', f)) & f.endswith('.mid')
             ]
lead_files.sort()

for i in range(len(lead_files)):
    lead_chords, lead_rhythm, lead_notes = read_midi.ProcessMidi(lead_files[i], 16)

    filename = f'{OUT_DIR}/encoded_song_{i}.txt'
    f = open(filename, "w")
    print(f'chord ct: {len(lead_chords)} rhythm ct: {len(lead_rhythm)} notes ct: {len(lead_notes)}')
    for j in range(len(lead_chords)-1):
        line = f'{lead_rhythm[j]} {lead_chords[j]} {lead_notes[j]}\n'
        f.write(line)
    f.close()
