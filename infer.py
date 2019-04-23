#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 08:57:23 2019

@author: azamatomu
"""
from model import EncoderBaseline, Classifier
import torch
from nltk.tokenize import word_tokenize

encoder = EncoderBaseline()
encoder.load_state_dict(torch.load('Baselineenc.pt'))
inp = 300
clf = Classifier(input_size = 4*inp,
                 classes = ["entailment", "contradiction", "neutral"],
                 hidden_size = 512, 
                 encoder = encoder, 
                 embedding = texts.vocab.vectors)

clf.load_state_dict(torch.load('Baseline.pt'), cpu=True)

premise = word_tokenize(premise)
hypothesis = word_tokenize(hypothesis)

premise = torch.tensor([texts.vocab.stoi[token] for token in premise]).to(DEVICE)
hypothesis = torch.tensor([texts.vocab.stoi[token] for token in hypothesis]).to(DEVICE)

# predict entailment
y_pred = clf.forward(
    (premise.expand(1, -1).transpose(0, 1), torch.tensor(len(premise)).to(DEVICE)),
    (hypothesis.expand(1, -1).transpose(0, 1), torch.tensor(len(hypothesis)).to(DEVICE))
)

# determine the type of inference
if y_pred.argmax().item() == 0:
    print('Entailment')
elif y_pred.argmax().item() == 1:
    print('Contradiction')
elif y_pred.argmax().item() == 2:
    print('Neutral')
else:
    raise ValueError('Invalid class!')
