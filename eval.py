# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

import sys
import io
import numpy as np
import logging
import os
import torch
# Set PATHs
PATH_TO_SENTEVAL = '/'
PATH_TO_DATA = os.getcwd() + '/data'
PATH_TO_VEC = 'small_glove_torchnlp.txt'
#PATH_TO_VEC = 'glove/glove.840B.300d.txt'
#PATH_TO_VEC = 'fasttext/crawl-300d-2M.vec'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


# SentEval prepare and batcher
def prepare(params, samples):
    
    #add special tokens to the dict.    
    samples.append(['<s>'])
    samples.append(['</s>'])
    samples.append(['<p>'])
    samples.append(['<unk>'])
    
    #initialize empity dictionaries
    words = {}
    word2id = {}

    for s in samples:
        for word in s:
            word2id[word] = text.vocab.stoi[word]
            words[word] = np.array(text.vocab.vectors[text.vocab.stoi[word]])
    
    params.word2id = word2id
    params.word_vec = words
    params.wvec_dim = 300
    
    #WTF this return nothing?
    return
def batcher(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []

    for sent in batch:
        sentvec = []
        for word in sent:
            if word in params.word_vec:
                sentvec.append(params.word_vec[word])
        if not sentvec:
            vec = np.zeros(params.wvec_dim)
            sentvec.append(vec)
        sentvec = np.mean(sentvec, 0)
        embeddings.append(sentvec)

    embeddings = np.vstack(embeddings)
    return embeddings


# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': False, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}

#params_senteval['infersent'] = encoder.to(DEVICE)

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
from model import EncoderBaseline, EncoderLSTM
from data import load_data, load_glove_model
if __name__ == "__main__":
    torch.device('cpu')
    
    #_, text, _ = load_data()
    word_embeddings = load_glove_model('small_glove_torchnlp.txt')
    text.build_vocab(all_data["dev"], vectors=word_embeddings)
    encoder = EncoderBaseline()
    #encoder = EncoderLSTM(emb_size=300, enc_size=2048, embedding=text.vocab.vectors)
    #text = texts
    #encoder.load_state_dict(torch.load('Baselineenc.pt'))
    #storage = torch.device('cpu')
    encoder.load_state_dict(torch.load('Baselineenc.pt',map_location=lambda storage, loc: storage))

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
                      'Length', 'WordContent', 'Depth', 'TopConstituents',
                      'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                      'OddManOut', 'CoordinationInversion']
    transfer_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'TREC', 'SST2',
                      'SICKEntailment']
    
    results = se.eval(transfer_tasks)
    print(results)