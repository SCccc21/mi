import torch, math
import numpy as np
from scipy.sparse import issparse, vstack

sen_list = [2, 3, 4, 5, 6, 7, 13, 14, 15]

def get_mask():
    mask = torch.zeros(1, 17).float().cuda()
    for idx in sen_list:
        mask[0, idx] = 1
    return mask

# max_norm 1
def trans_norm(x):
    return x / 8.07

def sklearn_invert(model, X, y, target_cols):
    assert X.shape[0] == y.shape[0] #check that X and y have compatible dimensions
    
    if issparse(X): #deal with sparse matrices correctly
        stack = vstack
    else:
        stack = np.stack
    
    guesses = []
    
    assert len(target_cols) > 0
    one_hot = (len(target_cols) > 1) #whether the target attribute was one-hot encoded (binary otherwise)
    num_variants = len(target_cols) if one_hot else 2 #number of possible values of the target
    
    for i in range(X.shape[0]): #iterate over the rows of X and y
        row_X = stack([X[i] for _ in range(num_variants)]) #create copies of X[i]
        if one_hot:
            row_X[:, target_cols] = np.eye(num_variants) #fill in with all possible values of target (one-hot encoded)
        else: #fill in with all possible values of target (binary)
            row_X[0, target_cols] = 0
            row_X[1, target_cols] = 1
        row_y = np.repeat(y[i], num_variants)
        
        errors = row_y - model.predict(trans_norm(row_X))
        scores = np.abs(errors)
        guesses.append(np.argmin(scores))
        
    return np.array(guesses)

def mlp_invert_continuous(model, X, y, target_col, model_name, min_val, max_val):
    model.eval()
    mask = get_mask()
    assert X.shape[0] == y.shape[0] #check that X and y have compatible dimensions
    
    guesses = []
    
    for i in range(X.shape[0]): #iterate over the rows of X and y
        val_list = np.linspace(min_val, max_val, num=100).tolist()
        x_list = []
        for possible_val in val_list:
            x_new = X[i, :]
            x_new[target_col] = possible_val
            x_new = torch.from_numpy(x_new).cuda().float().unsqueeze(0)
            x_list.append(x_new)

        x_try = torch.cat(x_list, dim=0)
        predicted, __, __ = model(x_try, x_try * mask)
        error = torch.abs(predicted - y[i])
        
        idx = torch.argmin(error).item()
        guesses.append(val_list[idx])


    return np.array(guesses)


def mlp_invert(model, X, y, target_cols, model_name):
    mask = get_mask()
    model.eval()
    assert X.shape[0] == y.shape[0] #check that X and y have compatible dimensions
    
    if issparse(X): #deal with sparse matrices correctly
        stack = vstack
    else:
        stack = np.stack
    
    guesses = []
    
    assert len(target_cols) > 0
    one_hot = (len(target_cols) > 1) #whether the target attribute was one-hot encoded (binary otherwise)
    num_variants = len(target_cols) if one_hot else 2 #number of possible values of the target
    
    for i in range(X.shape[0]): #iterate over the rows of X and y
        row_X = stack([X[i] for _ in range(num_variants)]) #create copies of X[i]
        if one_hot:
            row_X[:, target_cols] = np.eye(num_variants) #fill in with all possible values of target (one-hot encoded)
        else: #fill in with all possible values of target (binary)
            row_X[0, target_cols] = 0
            row_X[1, target_cols] = 1
        row_y = np.repeat(y[i], num_variants)

        
        row_X = torch.from_numpy(trans_norm(row_X)).float().cuda()
        
        if model_name == "reg" or model_name == "def" or model_name == "dp":
            errors = row_y - model(row_X).view(-1).cpu().detach().numpy()
        elif model_name == "vib":
            errors = row_y - model(row_X)[0].view(-1).cpu().detach().numpy()
        elif model_name == "sen":
            errors = row_y - model(row_X, row_X * mask)[0].view(-1).cpu().detach().numpy()

        scores = np.abs(errors)
        guesses.append(np.argmin(scores))
        
    return np.array(guesses)


