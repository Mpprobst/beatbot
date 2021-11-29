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
from shutil import copyfile
import torch
import mido

# project files
import agents
from agents import lstm_agent, nn_agent
import read_midi

TRAIN_DIR = "../data/train"  # reminder: train is split into leads and drums
TEST_DIR = "../data/test"  # reminder: train is split into leads and drums
OUT_DIR = "../output"
GRANULARITY = 16
NUM_MEASURES = 4
EPOCHS = 2

lead_files = [join(f'{TRAIN_DIR}/leads/',f)
              for f in listdir(f'{TRAIN_DIR}/leads')
                if isfile(join(f'{TRAIN_DIR}/leads/', f))
             ]
drum_folders = [join(f'{TRAIN_DIR}/drums/',f)
                for f in listdir(f'{TRAIN_DIR}/drums')
                    if isdir(join(f'{TRAIN_DIR}/drums/', f))
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

data = [] # inputs to the CNN

# prepare data
for i in range(len(lead_files)):
    lead_notes, lead_rhythm = read_midi.ProcessMidi(lead_files[i], GRANULARITY)
    _, drum_1 = read_midi.ProcessMidi(drum_files[i][0], GRANULARITY, True)
    _, drum_2 = read_midi.ProcessMidi(drum_files[i][1], GRANULARITY, True)
    #in_train = read_midi.ConcatMidiDatas(lead_notes, lead_rhythm)
    #in_val = read_midi.ConcatMidiDatas(drum_1, drum_2)
    in_train = lead_notes + lead_rhythm
    in_val = drum_1 + drum_2
    data.append((in_train, in_val))

# need to be able to switch out methods of ML
agent = lstm_agent.LSTMAgent(len(data[0][0]), 128)
for e in range(EPOCHS):
    for dat in data:
        agent.train(dat[0], dat[1])

# saving the trained model
#agent.save(OUT_DIR)

# test
testfiles = [join(f'{TEST_DIR}/',f)
              for f in listdir(f'{TEST_DIR}')
                if isfile(join(f'{TEST_DIR}/', f))
             ]

for i in range(len(testfiles)):
    test_notes, test_rhythm = read_midi.ProcessMidi(testfiles[i], GRANULARITY)
    #test = read_midi.ConcatMidiDatas(test_notes, test_rhythm)
    test = test_notes + test_rhythm
    out = agent.test(test)
    dir = join(OUT_DIR, f'test_{i}')
    if not isdir(dir):
        os.mkdir(dir)
    copyfile(testfiles[i], join(dir, os.path.basename(testfiles[i])))

    lead_midi = mido.MidiFile(testfiles[i])
    tpb = read_midi.GetTicksPerBeat(lead_midi)
    tempo = read_midi.GetTempo(lead_midi)
    ts = read_midi.GetTimeSig(lead_midi)
    #print(out)
    read_midi.CreateMidi(out, 2, f'{dir}/', tpb, tempo, ts, GRANULARITY)
