import torch
import numpy as np
import logging
import models, os
from work import *
from torch.autograd import Variable


# [cyp2c9, vkorc1]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
target_str = "vkorc1"
data_folder = 'data'
save_path = './checkpoint'

# [reg, vib]
model_name = 'reg'
att_epochs = 1000
att_lr = 2e-1
eps=0.4
t_val_min=-1
t_val_max=1

def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def main():
    # [0, 9, 16]
    global logger
    logger = get_logger()

    logger.info("=> load data ...")
    x, y, featnames = load_iwpc(data_folder)
    y = trans_project(y)

    t, target_cols = extract_target(x, target_str, featnames)

    target_model = models.MLP(input_dim=x.shape[1]).cuda()
    ckpt_name = './checkpoint/model_latest.pth'
    if os.path.isfile(ckpt_name):
        checkpoint = torch.load(ckpt_name)
        target_model.load_state_dict(checkpoint['state_dict'])
        logger.info("=> loaded target model checkpoint '{}'".format(ckpt_name))
    else:
        logger.info("=> no checkpoint found at '{}'".format(ckpt_name))

    logger.info("=> begin attacking ...")
    
    
    target_model.eval()

    if issparse(x): #deal with sparse matrices correctly
        stack = vstack
    else:
        stack = np.stack

    assert len(target_cols) > 0
    one_hot = (len(target_cols) > 1) #whether the target attribute was one-hot encoded (binary otherwise)
    logger.info("=> target attribute is one-hot? {}".format(one_hot))
    num_variants = len(target_cols) if one_hot else 2 #number of possible values of the targ
    guesses = []
    
    
    for i in range(x.shape[0]): #iterate over the rows of X and y
        row_x = stack([x[i] for _ in range(num_variants)]) #create copies of x[i]
        if one_hot:
            row_x[:, target_cols] = np.eye(num_variants) #fill in with all possible values of target (one-hot encoded)
        else: #fill in with all possible values of target (binary)
            row_x[0, target_cols] = 0
            row_x[1, target_cols] = 1
        
        row_y = np.repeat(y[i], num_variants)
        row_y = torch.from_numpy(row_y).float().cuda()
        row_x = torch.from_numpy(trans_norm(row_x)).float().cuda()
        
        Ipp = torch.eye(row_x.shape[1]).float().cuda()
        lam = 0.08

        row_x.requires_grad = True
        target_out = target_model(row_x).view(-1)
        losses = []
        # import pdb; pdb.set_trace()

        for idx in range(3):
            loss1 = (row_y[idx] - target_out[idx]).abs()
            loss1.backward(retain_graph=True)
            loss2 = 0
            for param in target_model.parameters():
                loss2 += param.grad
                
            loss = loss1 + lam * torch.sum(loss2.abs())
            losses.append(loss)

        losses = np.array(losses)
        guess = np.argmin(losses)
        guesses.append(guess)
        # import pdb; pdb.set_trace()
        
        print("person{}\t true:{}\t estimated:{}\t {}".format(i, t[i], guess, (guess==t[i])))

    # result = np.concatenate((t.unsqueeze(1), guesses.unsqueeze(1)), axis=1)
    # np.savetxt('result.csv', result)
    
    # attack acc
    num_correct = np.count_nonzero(guesses == t)
    num_rows = x.shape[0]
    attack_acc = num_correct / num_rows
    
    
    print("Attack Acc:{:.2f} ".format(attack_acc * 100))

    logger.info("=> Attack Finished.")


if __name__ == "__main__":
    main()