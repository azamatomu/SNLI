import numpy as np 
import os
import argparse

import torch
from torch import nn
from torch import optim
import torchtext
from torchtext import data
from model import EncoderBaseline, Classifier, EncoderLSTM, EncoderBiLSTM
from data import load_glove_model, load_data

if __name__ == "__main__":
    
    #Processing user input for selection of model
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_type', type=str, default='Baseline',
                        help='Baseline, LSTM, BiLSTM or MaxBiLSTM)')
    args = parser.parse_args()
    
    #Uploading and preprocessing the data
    glovename = "/small_glove_torchnlp.txt" #Small glove embeddings for words in SNLI
    glovedir = os.getcwd() + glovename
    word_embeddings = load_glove_model(glovedir)
    all_data, texts, labels = load_data() #Load data using torchtext Fields
    
    #Dataset used is SNLI fetched from the torchtext collection    
    texts.build_vocab(all_data["dev"], vectors=word_embeddings)
    labels.build_vocab(all_data["dev"])
    
    #Select encoder type
    inp = 2048 #input size is 2048 unless Baseline model used
    if args.encoder_type == 'Baseline':
        enc = EncoderBaseline()
        inp = 300
    if args.encoder_type == 'LSTM':
        enc = EncoderLSTM(emb_size=300, 
                          enc_size=2048, 
                          embedding=texts.vocab.vectors)
    if args.encoder_type == 'BiLSTM':        
        enc = EncoderBiLSTM(emb_size=300, 
                          enc_size=2048, 
                          batch_size=64,
                          embedding=texts.vocab.vectors,
                          pool_type=None)
    if args.encoder_type == 'MaxBiLSTM':        
        enc = EncoderBiLSTM(emb_size=300, 
                          enc_size=2048, 
                          batch_size=64,
                          embedding=texts.vocab.vectors,
                          pool_type='max')
    #Fully connected classifier layer
    clf = Classifier(input_size = 4*inp,
                     hidden_size = 512, 
                     classes = ["entailment", "contradiction", "neutral"],
                     encoder = enc, 
                     embedding = texts.vocab.vectors)
    #Define pars for training
    loss_func = nn.CrossEntropyLoss()
    lr = 0.1    
    opt = optim.SGD # or optim.Adam
    opt = opt(clf.parameters(), lr=lr, weight_decay=0)   
    epochs = 20
    
    #Switch to cuda if available
    if torch.cuda.is_available():
        loss_func = loss_func.to('cuda')
        clf = clf.to('cuda')    
    
    train_res = {'train_loss':[], 'val_loss': [], 'val_accuracy': []}
    print('Starting training')
    for epoch in range(1, epochs + 1):
        batch_size = 128
        train_iter, val_iter, test_iter = data.BucketIterator.splits(
                datasets=(all_data["train"], all_data["dev"], all_data["test"]), 
                batch_sizes=(batch_size, batch_size, batch_size),
                shuffle=True)
        
        clf.train() # turn on training mode
        running_loss = 0.0
        running_corrects = 0
        batchn = 0
        for batch in train_iter: 
            batchn += 1
            if batchn % 1000 == 0:
                print(f'Processed {batchn} batches')
            opt.zero_grad()      
            preds = clf(batch.premise, batch.hypothesis)
            loss = loss_func(preds,batch.label)
            loss.backward()
            opt.step()
            running_loss += loss.item() * batch_size

        clf.eval() # turn on evaluation mode
        val_loss = 0.0
        for batch in val_iter:
            preds = clf(batch.premise, batch.hypothesis)
            loss = loss_func(preds,batch.label)
            val_loss += loss.item() * batch_size
        accuracy = (batch.label == preds.argmax(dim=1)).float().mean().item()
        
        #Append training stats
        train_res['train_loss'].append(running_loss / len(all_data["train"]))        
        train_res['val_loss'].append(val_loss / len(all_data["dev"]))
        train_res['val_accuracy'].append(accuracy)
        #Update learning rate if necessary
        if epoch!=1:
            if (train_res['val_accuracy'][epoch-1] < train_res['val_accuracy'][epoch-2]):
                lr = lr / 5
                print(f'New learning rate: {lr}')
                for group in opt.param_groups:
                    group['lr'] = lr
        print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch, 
              train_res['train_loss'][epoch-1], train_res['val_loss'][epoch-1], train_res['val_accuracy'][epoch-1]))
        
        #Save checkpoints
        torch.save(clf.state_dict(), args.encoder_type + ".pt")
        torch.save(clf.encoder.state_dict(), args.encoder_type + "_enc.pt")
        np.save("results" + args.encoder_type, train_res)

        #Break loop if learning rate too small
        if lr < 1e-5:
            print('Finished training, at loss {lr}')
            break
