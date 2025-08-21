import os
import re
import json
import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils
import random
import pandas as pd
from transformers import BertModel, BertTokenizer
from sklearn.model_selection import train_test_split

folder_path = os.path.join(os.path.dirname(__file__), "Example")

def genData(file, max_len, seed=123):
    random.seed(seed)
    aa_dict = {aa: i + 1 for i, aa in enumerate("ARNDCQEGHILKMFPOSUTWYVX")}
    with open(file, 'r') as inf:
        lines = inf.read().splitlines()
    pep_codes, labels, pep_seq = [], [], []
    for pep in lines:
        pep, label = pep.split(",")
        labels.append(int(label))
        input_seq = ' '.join(pep)
        input_seq = re.sub(r"[UZOB]", "X", input_seq)
        pep_seq.append(input_seq)
        current_pep = [aa_dict[aa] for aa in pep]
        if len(current_pep) < max_len:
            padding_position = random.choice(['head', 'tail'])
            pad_length = max_len - len(current_pep)
            if padding_position == 'head':
                current_pep = [0] * pad_length + current_pep
            else:
                current_pep = current_pep + [0] * pad_length
        pep_codes.append(torch.tensor(current_pep))
    data = rnn_utils.pad_sequence(pep_codes, batch_first=True)
    return data, torch.tensor(labels), pep_seq

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False)
    bert = BertModel.from_pretrained("Rostlab/prot_bert_bfd") 

    for fold_no in range(1, 6):  
        train_file = f"{folder_path}/Train_fold_{fold_no}.csv"
        val_file = f"{folder_path}/Val_fold_{fold_no}.csv"

        train_data, train_label, train_seq = genData(train_file, 14, seed=123)
        val_data, val_label, val_seq = genData(val_file, 14, seed=123)

        seq = train_seq + val_seq

        seq2vec = {}
        for pep in seq:
            pep_str = "".join(pep)
            pep_text = tokenizer.tokenize(pep_str)
            pep_tokens = tokenizer.convert_tokens_to_ids(pep_text)
            tokens_tensor = torch.tensor([pep_tokens])
            with torch.no_grad():
                encoder_layers = bert(tokens_tensor)
                out_ten = torch.mean(encoder_layers.last_hidden_state, dim=1)
                out_ten = out_ten.numpy().tolist()
                seq2vec[pep] = out_ten

        with open(f'fold{fold_no}.emb', 'w') as g:
            g.write(json.dumps(seq2vec))
