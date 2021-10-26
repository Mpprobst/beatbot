import os
import read_midi
from os import listdir
from os.path import isfile, join, isdir
from mido import MidiFile, MidiTrack, Message

OUT_DIR = "../output"
out_folders = [join(f'{OUT_DIR}/',f)
                for f in listdir(f'{OUT_DIR}/')
                    if isdir(join(f'{OUT_DIR}/', f))
               ]
files = []
for folder in out_folders:
    files = [join(f'{folder}/',f)
             for f in listdir(f'{folder}/')
                if isfile(join(f'{folder}/',f))
            ]
    for f in files:
        mid = MidiFile(f, clip=True)
        print(mid)
