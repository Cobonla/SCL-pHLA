import os
import re
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import torch.nn.utils.rnn as rnn_utils
import time
import pickle
from termcolor import colored
from transformers import BertModel, BertTokenizer
import random
import pandas as pd
import torch.backends.cudnn as cudnn
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
# torch.cuda.set_device(2)
# os.environ['CUDA_LAUNCH_BLOCKING'] = '2'
import gc
import warnings
warnings.filterwarnings('ignore')

cudnn.benchmark = True
SEED = 0
print("Seed was: ", SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:4096"
#torch.cuda.set_per_process_memory_fraction(0.95)
gc.collect()
torch.cuda.empty_cache()
torch.cuda.set_device(0)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

class MyDataSet(Data.Dataset):
    def __init__(self, data, label, seq):
        self.data = data
        self.label = label
        self.seq = seq

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx], self.seq[idx]

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
        output, _ = self.gru(output)  #Gru
        output = output.permute(1, 0, 2)
        output = output.reshape(output.shape[0], -1)
        return self.block1(output)

    def trainModel(self, x, pep):
        with torch.no_grad():
            output = self.forward(x)
        outputs = torch.cat((output, pep), dim=1)
        del output
        return self.block2(outputs)

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) + (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

def collate(batch):
    seq1_ls, seq2_ls, label_ls, pep1_ls, pep2_ls = [], [], [], [], []
    label1_ls, label2_ls = [], []
    batch_size = len(batch)
    for i in range(int(batch_size / 2)):
        seq1, label1, pep_seq1 = batch[i][0], batch[i][1], batch[i][2]
        seq2, label2, pep_seq2 = batch[i + int(batch_size / 2)][0], batch[i + int(batch_size / 2)][1], batch[i + int(batch_size / 2)][2]
        label1_ls.append(label1.unsqueeze(0))
        label2_ls.append(label2.unsqueeze(0))
        pep1_ls.append(pep_seq1)
        pep2_ls.append(pep_seq2)
        label = (label1 ^ label2)
        seq1_ls.append(seq1.unsqueeze(0))
        seq2_ls.append(seq2.unsqueeze(0))
        label_ls.append(label.unsqueeze(0))
    seq1 = torch.cat(seq1_ls) #.to(device)
    seq2 = torch.cat(seq2_ls)# .to(device)
    label = torch.cat(label_ls)# .to(device)
    label1 = torch.cat(label1_ls)# .to(device)
    label2 = torch.cat(label2_ls)# .to(device)
    return seq1, seq2, label, label1, label2, pep1_ls, pep2_ls

def get_prelabel(data_iter, net):
    prelabel, relabel = [], []
    for x, y, z in data_iter:
        x, y = x.to(device), y.to(device)
        for i in range(len(z)):
            if i == 0:
                vec = torch.tensor(seq2vec[z[0]])
                vec = vec.to(device)
            else:
                temp_vec = torch.tensor(seq2vec[z[i]])
                temp_vec = temp_vec.to(device)
                vec = torch.cat((vec, temp_vec), dim=0)
                temp_vec.detach()
                del temp_vec
        outputs = net.trainModel(x, vec)
        vec.detach(), x.detach()
        del vec, x
        prelabel.append(outputs.argmax(dim=1).cpu().numpy())  #Finding the index of the max value along dimension 1
        relabel.append(y.cpu().numpy())
        y.detach(), outputs.detach()
        del y, outputs
        
        gc.collect()
        torch.cuda.empty_cache()
    del net
    gc.collect()
    torch.cuda.empty_cache()
    return prelabel, relabel


def calculate_metric(pred_y, labels, pred_prob):
    labels = np.array(labels, dtype=int)
    pred_prob = np.array(pred_prob, dtype=float)

    test_num = len(labels)
    tp = sum(int(labels[i] == 1 and labels[i] == pred_y[i]) for i in range(test_num))
    fp = sum(int(labels[i] == 0 and labels[i] != pred_y[i]) for i in range(test_num))
    tn = sum(int(labels[i] == 0 and labels[i] == pred_y[i]) for i in range(test_num))
    fn = sum(int(labels[i] == 1 and labels[i] != pred_y[i]) for i in range(test_num))

    ACC = float(tp + tn) / test_num
    Precision = float(tp) / (tp + fp) if tp + fp != 0 else 0
    Recall = Sensitivity = float(tp) / (tp + fn) if tp + fn != 0 else 0
    Specificity = float(tn) / (tn + fp) if tn + fp != 0 else 0
    MCC = (float(tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))) if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) != 0 else 0
    F1 = 2 * Recall * Precision / (Recall + Precision) if Recall + Precision != 0 else 0

    fpr, tpr, _ = roc_curve(labels, pred_prob, pos_label=1)
    AUC = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(labels, pred_prob, pos_label=1)
    AP = average_precision_score(labels, pred_prob, average='macro', pos_label=1, sample_weight=None)

    metric = torch.tensor([ACC, Precision, Sensitivity, Specificity, F1, AUC, MCC])
    roc_data = [fpr, tpr, AUC]
    prc_data = [recall, precision, AP]
    return metric, roc_data, prc_data


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for x, y, z in data_iter:
        x, y = x.to(device), y.to(device)
        for i in range(len(z)):
            if i == 0:
                vec = torch.tensor(seq2vec[z[0]])
                vec = vec.to(device)
            else:
                temp_vec = torch.tensor(seq2vec[z[i]])
                temp_vec = temp_vec.to(device)
                vec = torch.cat((vec, temp_vec), dim=0)
                temp_vec.detach()
                del temp_vec
        outputs = net.trainModel(x, vec)
        vec.detach(), x.detach()
        del vec, x
        acc_sum += (outputs.argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
        y.detach(), outputs.detach()
        del y, outputs
        gc.collect()
        torch.cuda.empty_cache()
    del net
    gc.collect()
    torch.cuda.empty_cache()
    return acc_sum / n

###Hyperparam 
emb_dims = [768, 512, 256, 128, 64]
num_layers_list = [1, 2, 3]
batch_sizes = [768, 512, 256, 128, 64]
learning_rates = [5e-5, 1e-5, 5e-4, 4e-4, 1e-4, 5e-3, 1e-3]
num_heads = [4, 8]
epochs = [1500]

for fold in range(1,6):
    print(f"\Training on Fold {fold}...\n")
    #Load dataaset
    train_data, train_label, train_seq = genData(f"{folder_path}/Train_fold_{fold}.csv", 14)
    test_data, test_label, test_seq = genData(f"{folder_path}/Val_fold_{fold}.csv", 14)

    print(train_data.shape, train_label.shape)
    print(test_data.shape, test_label.shape)

    train_dataset = MyDataSet(train_data, train_label, train_seq)
    test_dataset = MyDataSet(test_data, test_label, test_seq)

    seq2vec = json.load(open(f'fold{fold}.emb'))  # Load pretrained embeddings
    for emb_dim in emb_dims:
        for num_layers in num_layers_list:
            for batch_size in batch_sizes:
                for nh in num_heads:
                    for lr in learning_rates:
                        for epoch_num in epochs:
                            train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                            test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                            train_iter_cont = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)           
                            
                            net = newModel(emb_dim=emb_dim, num_heads = nh, num_layers=num_layers).to(device)
                
                            optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=5e-4)
                            criterion = ContrastiveLoss()
                            criterion_model = nn.CrossEntropyLoss(reduction='sum')
                            
                            best_acc = 0                     
                            EPOCH = epoch_num 
                            for epoch in range(EPOCH):
                                loss_ls = []
                                loss1_ls = []
                                loss2_3_ls = []
                                
                                t0 = time.time()
                                net.train()
                                gc.collect()
                                torch.cuda.empty_cache()
                        
                                for seq1, seq2, label, label1, label2, pep1, pep2 in train_iter_cont:
                                    for i in range(len(pep1)):
                                        if i == 0:
                                            pep1_2vec = torch.tensor(seq2vec[pep1[0]])
                                            pep1_2vec = pep1_2vec.to(device)
                                            pep2_2vec = torch.tensor(seq2vec[pep2[0]])
                                            pep2_2vec = pep2_2vec.to(device)
                                        else:
                                            temp_pep1_2vec = torch.tensor(seq2vec[pep1[0]])
                                            temp_pep1_2vec = temp_pep1_2vec.to(device)
                                            pep1_2vec = torch.cat((pep1_2vec,temp_pep1_2vec), dim=0)
                                            temp_pep2_2vec = torch.tensor(seq2vec[pep2[0]])
                                            temp_pep2_2vec = temp_pep2_2vec.to(device)
                                            pep2_2vec = torch.cat((pep2_2vec,temp_pep2_2vec), dim=0)
                                            temp_pep1_2vec.detach(), temp_pep2_2vec.detach()
                                            del temp_pep1_2vec, temp_pep2_2vec
                                    seq1 = seq1.to(device)
                                    seq2 = seq2.to(device)
                                    label1 = label1.to(device)
                                    label2 = label2.to(device)
                                    label = label.to(device)
                                    output1 = net(seq1)
                                    output2 = net(seq2)
                                    output3 = net.trainModel(seq1, pep1_2vec)
                                    output4 = net.trainModel(seq2, pep2_2vec)
                                    seq1.detach(), seq1.detach(), pep1_2vec.detach(), pep2_2vec.detach()
                                    del seq1, seq2, pep1_2vec, pep2_2vec
                                    gc.collect()
                                    torch.cuda.empty_cache()
                                    loss1 = criterion(output1, output2, label)
                                    loss2 = criterion_model(output3, label1)
                                    loss3 = criterion_model(output4, label2)
                                    loss = loss1 + loss2 + loss3
                                    optimizer.zero_grad()
                                    loss.backward()
                                    optimizer.step()
                                    
                                    loss_ls.append(loss.item())
                                    loss1_ls.append(loss1.item())
                                    loss2_3_ls.append((loss2 + loss3).item())
                            
                                    output1.detach(), output2.detach(), output3.detach(), output4.detach()
                                    loss.detach(), loss1.detach(), loss2.detach(), loss3.detach()
                                    label.detach(), label1.detach(), label2.detach()
                                del output1, output2, output3, output4, loss1, loss2, loss3, label1, label2, label
                                gc.collect()
                                torch.cuda.empty_cache()
                        
                                net.eval()
                                temp_model = net
                                with torch.no_grad():
                                    train_acc = evaluate_accuracy(train_iter, temp_model)
                                    test_acc = evaluate_accuracy(test_iter, temp_model)
                                    A, B = get_prelabel(test_iter, temp_model)
                                    A = [np.concatenate(A)]
                                    B = [np.concatenate(B)]
                                    A = np.array(A)
                                    B = np.array(B)
                                    A = A.reshape(-1, 1)
                                    B = B.reshape(-1, 1)
                                    df1 = pd.DataFrame(A, columns=['prelabel'])
                                    df2 = pd.DataFrame(B, columns=['realabel'])
                                    df4 = pd.concat([df1, df2], axis=1)
                                    del A, B
                        
                                    acc_sum, n = 0.0, 0
                                    outputs = []
                                    for x, y, z in test_iter:
                                        x, y = x.to(device), y.to(device)
                                        for i in range(len(z)):
                                            if i == 0:
                                                vec = torch.tensor(seq2vec[z[0]])
                                                vec = vec.to(device)
                                            else:
                                                temp_vec = torch.tensor(seq2vec[z[i]])
                                                temp_vec = temp_vec.to(device)
                                                vec = torch.cat((vec, temp_vec), dim=0)
                                                temp_vec.detach()
                                                del temp_vec
                                        output = torch.softmax(temp_model.trainModel(x, vec), dim=1)
                                        vec.detach(), x.detach(), y.detach()
                                        del vec, x, y
                                        gc.collect()
                                        torch.cuda.empty_cache()
                                        outputs.append(output)
                                    outputs = torch.cat(outputs, dim=0)
                                    pre_pro = outputs[:, 1]
                                    pre_pro = np.array(pre_pro.cpu().detach().numpy())
                                    pre_pro = pre_pro.reshape(-1)
                                    df3 = pd.DataFrame(pre_pro, columns=['pre_pro'])
                                    df5 = pd.concat([df4, df3], axis=1)
                                    real1 = df5['realabel']
                                    pre1 = df5['prelabel']
                                    pred_pro1 = df5['pre_pro']
                                    metric1, roc_data1, prc_data1 = calculate_metric(pre1, real1, pred_pro1)
                        
                                del outputs, output, pre_pro, temp_model
                                gc.collect()
                                torch.cuda.empty_cache()

                                if test_acc > best_acc:
                                    best_acc = test_acc
                                    with open(f'CL_fold{fold}.txt', 'a+') as f:
                                        f.write(f"best_acc: {best_acc}, metrics: {metric1}, emb_dim: {emb_dim}, num_layers: {num_layers}, batch_size: {batch_size}, num_heads: {nh}, learning_rate: {lr}, epoch: {epoch + 1}\n")
                                        print(f"best_acc: {best_acc}, metric: {metric1}, emb_dim: {emb_dim}, num_layers: {num_layers}, batch_size: {batch_size}, num_heads: {nh}, learning_rate: {lr}, epoch: {epoch + 1}")
                                    del test_acc
                                    gc.collect()
                                    torch.cuda.empty_cache()
