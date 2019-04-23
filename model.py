import numpy as np 
import torch
import torch.nn as nn


class EncoderBaseline(nn.Module):
    def __init__(self):
        super(EncoderBaseline, self).__init__()
    
    def forward(self, sent, sent_l):
        output = torch.div(torch.sum(sent, dim=0), sent_l.view(-1, 1).to(torch.float))

        return output

class EncoderLSTM(nn.Module):
    def __init__(self, emb_size, enc_size, embedding):
        super(EncoderLSTM, self).__init__()
        self.emb_size = emb_size
        self.enc_size = enc_size
        self.embedding = embedding
        self.enc = nn.LSTM(self.emb_size, self.enc_size, num_layers=1,
                           bidirectional=False)

    def forward(self, sent, sent_l):    
        sent_l, id_sort = np.sort(sent_l)[::-1], np.argsort(-sent_l)
        sent = sent.index_select(1, id_sort)
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_l.copy())
        output, _ = self.enc(sent_packed)[1][0].squeeze(0)
        id_unsort = np.argsort(id_sort)
        output = (output[0]).index_select(0, id_unsort)
        
        return output
    
class EncoderBiLSTM(nn.Module):
    def __init__(self, emb_size, enc_size, batch_size, embedding, pool_type=None):
        super(EncoderBiLSTM, self).__init__()
        self.batch_size = batch_size
        self.emb_size = emb_size
        self.enc_size = enc_size
        self.embedding = embedding
        self.output_size = 2*enc_size
        self.pool_type = pool_type
        self.enc = nn.LSTM(self.emb_size, self.enc_size, num_layers=1,
                           bidirectional=True)
        self.proj_enc = nn.Linear(self.output_size, self.output_size, bias=False)
        
    def forward(self, sent, sent_l):    
        sent_l, id_sort = np.sort(sent_l)[::-1], np.argsort(-sent_l)
        
        id_unsort = np.argsort(id_sort)
        sent = sent.index_select(1, id_sort)
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_l.copy())
        output, _ = self.enc(sent_packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)
        output = output[0].index_select(1, id_unsort)
        
        output = self.proj_enc(output.view(-1, self.output_size)).view(-1, 1, self.output_size)
        if self.pool_type is not None:
            output = torch.max(output, 0)[0].squeeze(0)
        return output
        
class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, classes, encoder, embedding):
        super(Classifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.classes = len(classes)
        self.encoder = encoder
        self.embedding = nn.Embedding.from_pretrained(embedding)
        self.clf = nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size), #Input layer
                nn.Linear(self.hidden_size, self.classes), #Softmax layer
                )

    def forward(self, premise_batch, hypothesis_batch):
        pre_s = premise_batch[0]
        pre_l = premise_batch[1]
        hyp_s = hypothesis_batch[0]
        hyp_l = hypothesis_batch[1]
        u_embed = self.embedding(pre_s)
        v_embed = self.embedding(hyp_s)
        u_encode = self.encode(u_embed, pre_l)
        v_encode = self.encode(v_embed, hyp_l)
        features = self.concat_embed(u_encode, v_encode)
        out = self.clf(features)
        return out

    def concat_embed(self, u,v):
        concat = torch.cat((u, v, (u-v).abs(), u*v), dim=1)
        return concat
    
    def encode(self, s, sl):
        emb = self.encoder(s, sl)
        return emb

