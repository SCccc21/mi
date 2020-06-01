import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.datasets import load_svmlight_file
from torch.distributions.laplace import Laplace
from sklearn.preprocessing import scale
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn import linear_model
from scipy.sparse import issparse, vstack
from scipy import sparse
from math import sqrt, ceil, pi
from logprob import Logprob
from copy import deepcopy
import matplotlib.pyplot as plt

import inversion
import random
import models
import torch
import csv
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

dim = 17

#linear regression
reg_epochs = 1000
reg_lr = 2e-1
#vib
vib_epochs = 500
vib_lr = 2e-1
# [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
beta = 0.11
#defense 
def_epochs = 1000
def_lr = 0.15
# 1~10
lamda = 10
# dp
dp_epochs = 1000
dp_lr = 0.12
# 0.1~1
dp_eps = 10

random_seed = 2

val_list = [0, 9, 16]
sen_list = [2, 3, 4, 5, 6, 7, 13, 14, 15]

torch.manual_seed(random_seed) # cpu
torch.cuda.manual_seed(random_seed) #gpu
np.random.seed(random_seed) #numpy
random.seed(random_seed)

def reduce(net):
    for p in net.parameters():
        p.data = torch.round(p.data * 10) / 10

def get_multipliers(errors):
    zs = errors
    return np.array([Logprob(-z**2 / 2, True) for z in zs])

def get_empirical_error(model, X, y):
    model.fit(X, y)
    mse = mean_squared_error(y, model.predict(X))
    return sqrt(mse)

def get_mask():
    mask = torch.zeros(1, 17).float().cuda()
    for idx in sen_list:
        mask[0, idx] = 1
    return mask

# max_norm 1
def trans_norm(x):
    return x / 8.07
    
def get_cross_validation_error(model, X, y):
    errors = []
    splitter = KFold(n_splits=10, shuffle=True, random_state=4294967295) #2**32 - 1
    for train_indices, test_indices in splitter.split(X):
        train_X, train_y = X[train_indices], y[train_indices]
        test_X, test_y = X[test_indices], y[test_indices]
        model.fit(train_X, train_y)
        mse = mean_squared_error(test_y, model.predict(test_X))
        errors.append(mse)
    
    return sqrt(sum(errors) / len(errors))

def normalize(counts):
    total = counts.sum()
    freqs = [Logprob(val / total) for val in counts]
    return np.array(freqs)

def sklearn_train_tree(max_depth):
    return DecisionTreeRegressor(max_depth=max_depth)

def extract_target(X, target_str, featnames):
    target_cols = np.array([i for i, featname in enumerate(featnames) if featname.startswith(target_str)])
    num_target = len(target_cols)
    if len(target_cols) >= 2:
        t = []
        for i in range(X.shape[0]):
            for j in range(num_target):
                if X[i, target_cols[j]] == 1:
                    t.append(j)
        t = np.array(t)
    else:
        assert len(target_cols) == 1 #check that only one column corresponds to target_str
        
        t = X[:, target_cols]
        if sparse.issparse(t):
            t = t.todense()
        t = np.squeeze(np.array(t))
        
        vals, counts = np.unique(t, return_counts=True)
        #assert np.array_equal(np.unique(t), np.arange(2)) #check that the target attribute is binary
        dist = normalize(counts)
    
    return t, target_cols

def inver_continuous(model, model_name, X, y, t, target_cols, min_val, max_val):
    assert X.shape[0] == y.shape[0] == t.shape[0]
    num_rows = X.shape[0]
    results = inversion.mlp_invert_continuous(model, X, y, target_cols, model_name, min_val, max_val)
    num_correct = np.sum(np.abs(results - t))
    
    return num_correct / num_rows

def inver(model, model_name, X, y, t, target_cols):
    assert X.shape[0] == y.shape[0] == t.shape[0]
    num_rows = X.shape[0]
    
    if model_name == "reg" or model_name == "vib" or model_name == "def" or model_name == "dp" or model_name == "sen":
        results = inversion.mlp_invert(model, X, y, target_cols, model_name)
    else:
        results = inversion.sklearn_invert(model, X, y, target_cols)

    num_correct = np.count_nonzero(results == t)
    
    return num_correct / num_rows

def get_mutual_infomation(out):
    sigma = 1
    bs = out.size(0)
    out_t = (out.view(1, -1) - out) ** 2
    out_t = torch.exp(out_t * -0.5 / sigma / sigma)
    const = 1 / sqrt(2 * pi * sigma * sigma)
    out_t = torch.sum(out_t * const, dim=1)
    out_t = - torch.mean(torch.log(out_t))
    return out_t

def engine(model, model_name, train_x, test_x, train_y, test_y, train_t, target_cols):
    #train_num = ceil(train_x.shape[0] / batch_size)
    if model_name == "reg":
        optimizer = torch.optim.SGD(model.parameters(), lr=reg_lr)
        epochs = reg_epochs
    elif model_name == "vib" or model_name == "sen":
        optimizer = torch.optim.SGD(model.parameters(), lr=vib_lr)
        epochs = vib_epochs
    elif model_name == "def":
        optimizer = torch.optim.SGD(model.parameters(), lr=def_lr)
        epochs = def_epochs
    elif model_name == "dp":
        optimizer = torch.optim.SGD(model.parameters(), lr=dp_lr)
        epochs = dp_epochs

    test_in = torch.from_numpy(trans_norm(test_x)).float().cuda()
    test_label = torch.from_numpy(test_y).float().cuda().view(-1, 1)

    train_in = torch.from_numpy(trans_norm(train_x)).float().cuda()
    train_label = torch.from_numpy(train_y).float().cuda().view(-1, 1)
    tot = train_in.size(0)

    bs = train_label.size(0)
    mask = get_mask()

    best_error = 100
    best_model = None

    #train_in = train_in[:1, :]
    #train_label = train_label[:]

    for e in range(epochs):
        model.train()
        
        delta = 2*(1+dim)*(1+dim)
        num_phi_1 = dim
        num_phi_2 = dim ** 2
           
        input = train_in[:, :]
        label = train_label[:, :]

        if model_name == "reg":
            out = model(input)
            loss = torch.mean((label - out) ** 2)
        elif model_name == "vib":
            out, mu, std = model(input)
            reg_loss = torch.mean((label - out) ** 2)
            info_loss = - 0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(dim=1).mean()
            loss = reg_loss + beta * info_loss
            
        elif model_name == "sen":
            out, mu, std = model(input, input * mask)
            reg_loss = torch.mean((label - out) ** 2)
            info_loss = - 0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(dim=1).mean()
            loss = reg_loss + beta * info_loss
        
        elif model_name == "def":
            out = model(input)
            reg_loss = torch.mean((label - out) ** 2)
            info_loss = get_mutual_infomation(model(input * mask))
            loss = reg_loss + lamda * info_loss
            
        elif model_name == "dp":
            noise = Laplace(0, delta/dp_eps).sample().cuda()
            for w in model.parameters():
                lamda_1 = torch.sum(input * label, dim=0) * 2
                input_a = input.unsqueeze(1)
                input_b = input.unsqueeze(2)
                dot = input_a * input_b
                lamda_2 = torch.sum(dot, dim=0)
                lamda_1 += noise
                lamda_2 += noise
                loss1 = lamda_1 * w
                w_dot = w * w.view(-1, 1)
                loss2 = lamda_2 * w_dot
                loss = - torch.sum(loss1) + torch.sum(loss2)
                loss /= tot
                
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        if model_name == "reg" or model_name == "def" or model_name == "dp":
            test_out = model(test_in)
            train_out = model(train_in)
        elif model_name == "vib":
            test_out = model(test_in)[0]
            train_out = model(train_in)[0]
        elif model_name == "sen":
            test_out = model(test_in, test_in * mask)[0]
            train_out = model(train_in, train_in * mask)[0]

        test_error = torch.sqrt(torch.mean((test_out - test_label) ** 2)).item()
        train_error = torch.sqrt(torch.mean((train_out - train_label) ** 2)).item()
       
        if (e+1) % 10 == 0:
            attack_acc = inver(model, model_name, train_x, train_y, train_t, target_cols)
            print("Epoch:{}\tTrain Error:{:.4f}\tTest Error:{:.4f}\tAttack Acc:{:.2f}".format(e, train_error, test_error, attack_acc))

    #reduce(model)
    model.eval()
    if model_name == "reg" or model_name == "def" or model_name == "dp":
        test_out = model(test_in)
        train_out = model(train_in)
    elif model_name == "vib":
        test_out = model(test_in)[0]
        train_out = model(train_in)[0]
    elif model_name == "sen":
        test_out = model(test_in, test_in * mask)[0]
        train_out = model(train_in, train_in * mask)[0]

    test_error = torch.sqrt(torch.mean((test_out - test_label) ** 2)).item()
    train_error = torch.sqrt(torch.mean((train_out - train_label) ** 2)).item()
    print("Final Train Error:{:.4f}\tTest Error:{:.4f}".format(train_error, test_error))

    return train_error, test_error

def load_iwpc(data_folder):
    datafile = '{}/iwpc-scaled.csv'.format(data_folder)
    col_types = {'race': str,
                 'age': float,
                 'height': float,
                 'weight': float,
                 'amiodarone': int,
                 'decr': int,
                 'cyp2c9': str,
                 'vkorc1': str,
                 'dose': float}
    X, y = [], []
    with open(datafile) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for col_name in reader.fieldnames:
                col_type = col_types[col_name]
                row[col_name] = col_type(row[col_name]) #cast to correct type
                if col_name == 'dose':
                    y.append(row[col_name])
                    del row[col_name]
            X.append(row)
    
    dv = DictVectorizer()
    X = dv.fit_transform(X).toarray()
    y = np.array(y)
    featnames = np.array(dv.get_feature_names())
    return X, y, featnames

def load_iwpc_m(data_folder):
    datafile = '{}/iwpc-scaled.csv'.format(data_folder)
    col_types = {'race': str,
                 'age': float,
                 'height': float,
                 'weight': float,
                 'amiodarone': int,
                 'decr': int,
                 'cyp2c9': str,
                 'vkorc1': str,
                 'dose': float}
    X, y = [], []
    with open(datafile) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for col_name in reader.fieldnames:
                col_type = col_types[col_name]
                row[col_name] = col_type(row[col_name]) #cast to correct type
                if col_name == 'dose':
                    y.append(row[col_name])
                    del row[col_name]
            X.append(row)
    
    dv = DictVectorizer()
    X = dv.fit_transform(X).toarray()
    y = np.array(y)
    featnames = np.array(dv.get_feature_names())
    return X, y, featnames
############ 
############

# [-1, 1]
def trans_project(y):
    min_y, max_y = np.min(y[:]), np.max(y[:])
    y = (y - min_y) / (max_y - min_y)
    y = (y - 0.5) * 2
    return y

def trans_project_t(t):
    min_t, max_t = torch.min(t[:]), torch.max(t[:])
    t = (t - min_t) / (max_t - min_t)
    t = (t - 0.5) * 2
    return t

# [0,2]
def inver_project(t):
    t = t / 2 + 0.5
    t = t * 2
    return t

def merge_t(t, x, target_cols):
    t = inver_project(t)
    for i in range(x.shape[0]):
        if t[i] >= 1.5:
            t[i] = 2
        elif t[i] >= 0.5 and t[i]<1.5:
            t[i] = 1
        else:
            t[i] = 0

        for j in range(len(target_cols)):
            x[i, target_cols[j]] = 1 if j == t[i] else 0

        return x

def get_result(x, t, target_cols):
    pred_t = []
    for i in range(x.shape[0]):
        vmax = torch.max(x[i, target_cols])
        for j in range(3):
            if x[i, target_cols[j]] == vmax:
                pred_t.append(j)
                break
        # if i>4600:
        #     print(i,j)
    
    pred_t = np.array(pred_t)

    # attack acc
    num_correct = np.count_nonzero(pred_t == t)
    num_rows = x.shape[0]
    attack_acc = num_correct / num_rows
    
    return pred_t, attack_acc
