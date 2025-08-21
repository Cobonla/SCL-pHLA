import os
import re
import random
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.utils.rnn as rnn_utils
import torch.backends.cudnn as cudnn
import pandas as pd
import numpy as np
import json
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score, auc

torch.backends.cudnn.benchmark = True
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.cuda.set_device(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

##Path
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Final_models")
folder_path = os.path.join(os.path.dirname(__file__), "Example")

def genData(file, max_len, seed=123):
    random.seed(seed)
    aa_dict = {aa: i + 1 for i, aa in enumerate("ARNDCQEGHILKMFPOSUTWYVX")}
    
    with open(file, 'r') as inf:
        lines = inf.read().splitlines()
    
    pep_codes, pep_seq = [], []
    for pep in lines:
        pep = pep.strip()  
        input_seq = re.sub(r"[UZOB]", "X", pep)  
        pep_seq.append(input_seq)
        
        current_pep = [aa_dict[aa] for aa in list(input_seq)]

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

class MyDataSet(Data.Dataset):
    def __init__(self, data, seq):
        self.data = data
        self.seq = seq

    def __getitem__(self, idx):
        return self.data[idx], self.seq[idx]

    def __len__(self):
        return len(self.data) 

class newModel(nn.Module):
    def __init__(self, vocab_size=26, emb_dim=512, num_heads=8, num_layers=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = 25
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.embedding = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=0)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.emb_dim, nhead=self.num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)
        self.gru = nn.GRU(self.emb_dim, self.hidden_dim, num_layers=self.num_layers+1, bidirectional=True, dropout=0.5)
        
        self.block1 = nn.Sequential(nn.Linear(14*self.hidden_dim*2, 2048),
                                    nn.BatchNorm1d(2048),
                                    nn.LeakyReLU(),
                                    nn.Linear(2048, 1024))
        
        self.block2 = nn.Sequential(
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 640),
            nn.BatchNorm1d(640),
            nn.LeakyReLU(),
            nn.Linear(640, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 2))

    def forward(self, x):
        x = self.embedding(x)
        output = self.transformer_encoder(x).permute(1, 0, 2)
        output, _ = self.gru(output)  # GRU
        output = output.permute(1, 0, 2)
        output = output.reshape(output.shape[0], -1)
        return self.block1(output)

    def trainModel(self, x, pep):
        with torch.no_grad():
            output = self.forward(x)
        outputs = torch.cat((output, pep), dim=1)
        del output
        return self.block2(outputs)

def test_predict():
    #Load model params 
    ##Example for HLA-C*17:01
    emb_dim = 512
    num_layers = 1
    num_heads = 8
    batch_size = 64

    checkpoint = torch.load(os.path.join(model_path, "HLA-C1701", "HLA-C*17:01.pl"), map_location=device)  
    best_model_state = checkpoint['model']

    net = newModel(emb_dim=emb_dim, num_heads=num_heads, num_layers=num_layers).to(device)
    net.load_state_dict(best_model_state)
    net.eval()

    # Prepare data loader
    test_data, test_seq = genData(os.path.join(folder_path, "test.csv"), 14)
    print(test_data.shape)

    test_dataset = MyDataSet(test_data, test_seq)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    seq2vec = json.load(open(os.path.join(folder_path, "test.emb")))

    # Predict and save results
    predictions = []

    with torch.no_grad():
        for x, seq in test_iter:
            x = x.to(device)
            vec = torch.stack([torch.tensor(seq2vec[s]) for s in seq]).squeeze(1).to(device)  
            output = torch.softmax(net.trainModel(x, vec), dim=1)
            probabilities = output[:, 1].cpu().numpy()  

            for pep, prob in zip(seq, probabilities):
                class_label = "Binder" if prob > 0.5 else "Non-binder"
                predictions.append([pep, class_label, round(prob, 4)])

    df = pd.DataFrame(predictions, columns=["Name", "Class", "Probability"])
    df.to_csv("prediction_result.csv", index=False, sep=",", quoting=3)

if __name__ == "__main__":
    test_predict()

