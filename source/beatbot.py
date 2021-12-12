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
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import argparse

# project files
import agents
from agents import lstm_agent, nn_agent
import read_midi

TRAIN_DIR = "../data/train"  # reminder: train is split into leads and drums
TEST_DIR = "../data/test"  # reminder: train is split into leads and drums
OUT_DIR = "../output"
GRANULARITY = 16
NUM_MEASURES = 4
EPOCHS = 50  #100 is good for nn
BATCH_SIZE = 2
MODEL_TYPES = {"nn" : 0, "lstm" : 1}

def pad_input(sequence, length):
    features = np.zeros((len(sequence), length), dtype=int)
    for ii, review in enumerate(sequence):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:length]
    return features

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default = "nn", help='Define type of nn model (nn or lstm)')
args = parser.parse_args()

agent_type = 0
if args.model in MODEL_TYPES.keys():
    agent_type = MODEL_TYPES[args.model]
else:
    print(f'model type {args.model} invalid')
    exit()

print(f'model type {agent_type} : {args.model}')
lead_files = [join(f'{TRAIN_DIR}/leads/',f)
              for f in listdir(f'{TRAIN_DIR}/leads')
                if isfile(join(f'{TRAIN_DIR}/leads/', f))
             ]
drum_folders = [join(f'{TRAIN_DIR}/drums/',f)
                for f in listdir(f'{TRAIN_DIR}/drums')
                    if isdir(join(f'{TRAIN_DIR}/drums/', f))
               ]
test_files = [join(f'{TEST_DIR}/',f)
              for f in listdir(f'{TEST_DIR}')
                if isfile(join(f'{TEST_DIR}/', f))
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

train_data = []
train_lab = []
test_data = []

num_features = 300
seq_len = 0

# prepare data
for i in range(len(lead_files)):
    lead_notes, lead_rhythm = read_midi.ProcessMidi(lead_files[i], GRANULARITY)
    _, drum_1 = read_midi.ProcessMidi(drum_files[i][0], GRANULARITY, True)
    _, drum_2 = read_midi.ProcessMidi(drum_files[i][1], GRANULARITY, True)

    in_train = lead_notes + lead_rhythm
    in_val = drum_1 + drum_2

    train_data.append(in_train)
    train_lab.append(in_val)
    seq_len = len(in_train)

train_data = pad_input(train_data, seq_len)
train_lab = pad_input(train_lab, seq_len)
test_data = pad_input(test_data, seq_len)
train_data = torch.Tensor(train_data)
train_lab = torch.Tensor(train_lab)
train_data = TensorDataset(train_data, train_lab)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE)

#train
agent = None
if agent_type == 0:
    agent = nn_agent.NNAgent(len(train_data[0][0]), 128)
elif agent_type == 1:
    agent = lstm_agent.LSTMAgent(num_features, seq_len, 64)

for e in range(EPOCHS):
    print(f'epoch: {e} loss: {agent.running_loss}')
    if agent_type == 1:
        h = agent.model.init_hidden(BATCH_SIZE)    #lstm only
    for dat, lab in train_loader:
        #print(f'data: {dat.size()}{dat}')
        if agent_type == 0:             #nn
            agent.train(dat, lab)
        elif agent_type == 1:           #lstm
            agent.train(dat, lab, h)

# saving the trained model
#agent.save(OUT_DIR)

# test
if agent_type == 1:
    agent.model.eval()  #lstm only?
    h = agent.model.init_hidden(BATCH_SIZE)
batch = []
file_no = 0
for i in range(len(test_files)):
    test_notes, test_rhythm = read_midi.ProcessMidi(test_files[i], GRANULARITY)
    test = test_notes + test_rhythm
    if agent_type == 0:
        out = agent.test(test)
        dir = join(OUT_DIR, f'test_{i}')
        if not isdir(dir):
            os.mkdir(dir)
        copyfile(test_files[i], join(dir, os.path.basename(test_files[i])))

        lead_midi = mido.MidiFile(test_files[i])
        tpb = read_midi.GetTicksPerBeat(lead_midi)
        tempo = read_midi.GetTempo(lead_midi)
        ts = read_midi.GetTimeSig(lead_midi)
        #print(out)
        read_midi.CreateMidi(out, 2, f'{dir}/', tpb, tempo, ts, GRANULARITY)
    elif agent_type == 1:
        batch.append(test)
        if len(batch) >= BATCH_SIZE:
            #out = agent.test(test)
            batch = np.array(batch)
            out, h = agent.test(batch, h)

            for j in range(BATCH_SIZE):
                dir = join(OUT_DIR, f'test_{file_no}')
                if not isdir(dir):
                    os.mkdir(dir)
                copyfile(test_files[file_no], join(dir, os.path.basename(test_files[file_no])))

                lead_midi = mido.MidiFile(test_files[file_no])
                tpb = read_midi.GetTicksPerBeat(lead_midi)
                tempo = read_midi.GetTempo(lead_midi)
                ts = read_midi.GetTimeSig(lead_midi)
                print(f'Creating {file_no}')
                read_midi.CreateMidi(out[j], 2, f'{dir}/', tpb, tempo, ts, GRANULARITY)
                file_no += 1
            batch = []
