# SNLI
Practical assignment 1 for the Statistical Methods for Natural Language Semantics course, University of Amsterdam.

## Scripts
python train.py `--encoder_type`

`--encoder_type`: Baseline, LSTM, BiLSTM or MaxBiLSTM

This script trains a model for a given encoder type.
##
python eval.py `--encoder_type` `--emb_file`

`--encoder_type`: Baseline, LSTM, BiLSTM or MaxBiLSTM
`--emb_file`:     Word Vectors

This script runs evaluation on SNLI and SentEval sets for a given encoder type.
##
python infer.py `--encoder_type`

`--encoder_type`: Baseline, LSTM, BiLSTM or MaxBiLSTM

This script shows an inference experiment for a given encoder_type.
Given remise:     `A boy was outside and he was playing with friends`
Given hypotheses: `[A boy was inside], [A boy has no friends], [It is dark]`

##

Works consulted in the process:

https://github.com/facebookresearch/SentEval

https://github.com/facebookresearch/InferSent/
