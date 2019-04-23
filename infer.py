from model import EncoderBaseline, EncoderLSTM, EncoderBiLSTM, Classifier
import torch
import argparse
from nltk.tokenize import word_tokenize

from data import load_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_type', type=str, default='Baseline',
                        help='Baseline, LSTM, BiLSTM or MaxBiLSTM)')
    args = parser.parse_args()
    _, text, _ = load_data()
    if torch.cuda.is_available():
        torch.device('cuda')
    else:
        torch.device('cpu')
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

    labels = ['entailment', 'contradiction', 'neutral']
    hypotheses = ['A boy was inside', 'A boy has no friends', 'It is bright']
    
    for hypothesis in hypotheses:
      premise = 'A boy was outside and he was playing with friends'
      print(premise)
      print(hypothesis)
      premise = word_tokenize(premise)
      hypothesis = word_tokenize(hypothesis)
      
      premise = torch.tensor([text.vocab.stoi[token] for token in premise])
      hypothesis = torch.tensor([text.vocab.stoi[token] for token in hypothesis])
      
      premise_l = torch.tensor(len(premise))
      hypothesis_l = torch.tensor(len(hypothesis))
      
      if torch.cuda.is_available:
          premise = premise.to('cuda')
          hypothesis = hypothesis.to('cuda')
          premise_l = premise_l.to('cuda')
          hypothesis_l = hypothesis_l.to('cuda')      
          
      y_pred = clf.forward(
          (premise.expand(1, -1).transpose(0, 1).cpu(), premise_l.cpu()),
          (hypothesis.expand(1, -1).transpose(0, 1).cpu(), hypothesis_l.cpu())
      )
      print(labels[y_pred.argmax().item()])
