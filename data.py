import numpy as np 
import json
import os
from nltk.tokenize import word_tokenize

import torch
import torchtext
from torch import nn
from torch import optim
from torchtext import data, datasets, vocab
from model import EncoderBaseline, Classifier, EncoderLSTM, EncoderBiLSTM

def load_glove_model(glovedir):
    print("Loading Glove Model")
    model = torchtext.vocab.Vectors(glovedir)
    print("Done.")
    return model

def load_data():
    #all_data = {"train": None, "dev": None, "test": None}
    x_field = data.Field(lower=True,
                 tokenize=word_tokenize,
                 include_lengths=True)
    y_field = data.Field(sequential=False,
                 pad_token=None,
                 unk_token=None,
                 is_target=True)
    all_data = {}
    all_data["train"], all_data["dev"], all_data["test"]= datasets.SNLI.splits(x_field, y_field)
    return all_data, x_field, y_field

if __name__ == "__main__":
    
    glovename = "/small_glove_torchnlp.txt"
    glovedir = os.getcwd() + glovename
    word_embeddings = load_glove_model(glovedir)
    all_data, texts, labels = load_data()
    texts.build_vocab(all_data["dev"], vectors=word_embeddings)
    labels.build_vocab(all_data["dev"])
    #%%    
    batch_size = 64
    dev_iter, test_iter = data.BucketIterator.splits(
                datasets=(all_data["dev"], all_data["test"]), 
                batch_sizes=(batch_size, batch_size), 
                shuffle=True)
    
    
    #%%
    enc = EncoderBaseline()
    inp = 300
    #%%
    enc = EncoderLSTM(emb_size=300, 
                      enc_size=2048, 
                      embedding=texts.vocab.vectors)
    inp = 2048
    enc = EncoderBiLSTM(emb_size=300, 
                      enc_size=2048, 
                      batch_size=64,
                      embedding=texts.vocab.vectors,
                      pool_type=None)
    inp = 2048
    
    clf = Classifier(input_size = 4*inp,
                     hidden_size = 512, 
                     classes = ["entailment", "contradiction", "neutral"],
                     encoder = enc, 
                     embedding = texts.vocab.vectors)
    weight = torch.FloatTensor(clf.classes).fill_(1)
    loss_func = nn.CrossEntropyLoss(weight=weight)
    loss_func.size_average = False
    
    lr = 0.1    
    opt = optim.SGD # or optim.Adam
    opt = opt(clf.parameters(), lr=lr)   
    epochs = 10
    train_res = {'train_loss':[], 'val_loss': [], 'val_accuracy': []}
     
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        running_corrects = 0
        clf.train() # turn on training mode
        for batch in dev_iter: # thanks to our wrapper, we can intuitively iterate over our data!
            opt.zero_grad()      
            preds = clf(batch.premise, batch.hypothesis)
            loss = loss_func(preds,batch.label)
            loss.backward()
            opt.step()
            running_loss += loss.item() * batch_size
        train_res['train_loss'].append(running_loss / len(all_data["dev"]))        
        # calculate the validation loss for this epoch
        val_loss = 0.0
        clf.eval() # turn on evaluation mode
        for batch in test_iter:
            preds = clf(batch.premise, batch.hypothesis)
            loss = loss_func(preds,batch.label)
            val_loss += loss.item() * batch_size
        train_res['val_loss'].append(val_loss / len(all_data["test"]))
        accuracy = (batch.label == preds.argmax(dim=1)).float().mean().item()
        train_res['val_accuracy'].append(accuracy)
        if epoch!=1:
            if (train_res['val_accuracy'][epoch-1] < train_res['val_accuracy'][epoch-2]):
                lr = lr / 5
                print(f'New learning rate: {lr}')
                for group in opt.param_groups:
                    group['lr'] = lr
        print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch, 
              train_res['train_loss'][epoch-1], train_res['val_loss'][epoch-1], train_res['val_accuracy'][epoch-1]))
        if lr < 1e-5:
            print('Finished training, at loss {lr}')
            break
    
        
