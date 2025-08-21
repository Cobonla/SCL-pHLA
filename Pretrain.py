import os
import re
import json
import torch
import torch.nn.utils.rnn as rnn_utils
import random
from transformers import BertModel, BertTokenizer

# Path
folder_path = os.path.join(os.path.dirname(__file__), "Example")
test_file = os.path.join(folder_path, "test.csv")
output_file = os.path.join(folder_path, "test.emb")

def genData(file, max_len, seed=123):
    random.seed(seed)
    aa_dict = {aa: i + 1 for i, aa in enumerate("ARNDCQEGHILKMFPOSUTWYVX")}
    with open(file, 'r') as inf:
        lines = [line.strip() for line in inf if line.strip()]
    pep_codes, pep_seq = [], []
    for pep in lines:
        input_seq = re.sub(r"[UZOB]", "X", pep)
        pep_seq.append(input_seq)
        
        current_pep = [aa_dict.get(aa, 0) for aa in list(input_seq)]

        if len(current_pep) < max_len:
            padding_position = random.choice(['head', 'tail'])
            pad_length = max_len - len(current_pep)
            if padding_position == 'head':
                current_pep = [0] * pad_length + current_pep
            else:
                current_pep = current_pep + [0] * pad_length

        pep_codes.append(torch.tensor(current_pep))
    data = rnn_utils.pad_sequence(pep_codes, batch_first=True)
    return data, pep_seq

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False)
    bert = BertModel.from_pretrained("Rostlab/prot_bert_bfd")
    test_data, test_seq = genData(test_file, 14, seed=123)
    seq2vec = {}
    for i, pep in enumerate(test_seq, 1):
        pep_text = tokenizer.tokenize(pep)
        pep_tokens = tokenizer.convert_tokens_to_ids(pep_text)
        tokens_tensor = torch.tensor([pep_tokens])

        with torch.no_grad():
            encoder_layers = bert(tokens_tensor)
            out_ten = torch.mean(encoder_layers.last_hidden_state, dim=1)
            out_ten = out_ten.cpu().numpy().tolist()
            seq2vec[pep] = out_ten

    with open(output_file, 'w') as g:
        g.write(json.dumps(seq2vec))
