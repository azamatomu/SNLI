from __future__ import absolute_import, division, unicode_literals

import sys
import numpy as np
import os
import torch
import io
from model import EncoderBaseline, EncoderLSTM, EncoderBiLSTM, Classifier
from data import load_data, load_glove_model
import argparse
import senteval
from torchtext import data
PATH_TO_SENTEVAL = '/'
PATH_TO_DATA = os.getcwd() + '/data'
sys.path.insert(0, PATH_TO_SENTEVAL)


# SentEval prepare and batcher
def prepare(params, samples):
    
    samples.append(['<s>'])
    samples.append(['</s>'])
    samples.append(['<p>'])
    samples.append(['<unk>'])
    
    words = {}
    word2id = {}

    for s in samples:
        for word in s:
            word2id[word] = text.vocab.stoi[word]
            words[word] = np.array(text.vocab.vectors[text.vocab.stoi[word]])
    
    params.word2id = word2id
    params.word_vec = words
    params.wvec_dim = 300
    return

def evaluate(all_data, clf):
    test_res = {'test_accuracy': []}
    batch_size = len(all_data["test"])
    _, _, test_iter = data.BucketIterator.splits(
        datasets=(all_data["train"], all_data["dev"], all_data["test"]), 
        batch_sizes=(batch_size, batch_size, batch_size),
        shuffle=True)
    clf.eval() # turn on evaluation mode
    preds = clf(test_iter.premise, test_iter.hypothesis)
    accuracy = (test_iter.label == preds.argmax(dim=1)).type(torch.float).mean().item()
    test_acc = accuracy/len(test_iter)
    test_res['val_accuracy'].append(test_acc)
    return test_res
    
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_file', type=str, default='small_glove',
                        help='Small or full Glove')
    parser.add_argument('--encoder_type', type=str, default='Baseline',
                        help='Baseline, LSTM, BiLSTM or MaxBiLSTM)')
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        torch.device('cuda')
    else:
        torch.device('cpu')
    
    all_data, text, _ = load_data()
    if args.emb_file == 'small_glove':
        glovedir = 'small_glove_torchnlp.txt'
    else:
        glovedir = 'glove.840B.300d.txt'
    word_embeddings = load_glove_model(glovedir)
    all_data, text, _ = load_data()
    text.build_vocab(all_data["train"], vectors=word_embeddings)
    
    inp = 2048
    if args.encoder_type == 'Baseline':
        encoder = EncoderBaseline()
        inp = 300
        encoder.load_state_dict(torch.load('Baselineenc.pt',map_location=lambda storage, loc: storage))
    if args.encoder_type == 'LSTM':
        encoder = EncoderLSTM(emb_size=300, 
                          enc_size=2048, 
                          embedding=text.vocab.vectors)
        encoder.load_state_dict(torch.load('LSTM_enc.pt',map_location=lambda storage, loc: storage))
    if args.encoder_type == 'BiLSTM':        
        encoder = EncoderBiLSTM(emb_size=300, 
                          enc_size=2048, 
                          batch_size=64,
                          embedding=text.vocab.vectors,
                          pool_type=None)
    if args.encoder_type == 'MaxBiLSTM':        
        encoder = EncoderBiLSTM(emb_size=300, 
                          enc_size=2048, 
                          batch_size=64,
                          embedding=text.vocab.vectors,
                          pool_type='max')

    clf = Classifier(input_size = 4*inp,
                 classes = ["entailment", "contradiction", "neutral"],
                 hidden_size = 512, 
                 encoder = encoder, 
                 embedding = text.vocab.vectors)

    if args.encoder_type == 'Baseline':
        clf.load_state_dict(torch.load('Baseline.pt',map_location=lambda storage, loc: storage))        
    if args.encoder_type == 'LSTM':
        clf.load_state_dict(torch.load('LSTM.pt',map_location=lambda storage, loc: storage))        

    print('Imported everything needed.')
    
    print('Starting evaluation on SentEval.')
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
                      'Length', 'WordContent', 'Depth', 'TopConstituents',
                      'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                      'OddManOut', 'CoordinationInversion']
    transfer_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'TREC', 'SST2','SICKEntailment']
    
    text_trap = io.StringIO()
    sys.stdout = text_trap
    results = se.eval(transfer_tasks)
    sys.stdout = sys.__stdout__
    print(results)
    np.save("senteval" + args.encoder_type, results)
    
    print('Starting evaluation on SNLI.')
    test_res = evaluate(all_data, clf)
    print(test_res)
    np.save("snli" + args.encoder_type, results)
    
    

    
