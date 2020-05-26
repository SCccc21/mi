from work import load_iwpc, extract_target, inver, inver_continuous, engine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from math import sqrt
import numpy as np
import models, os
import logging
import torch
from work import *

# [cyp2c9, vkorc1]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

target_str = "vkorc1"
data_folder = 'data'
save_path = './checkpoint'
# [reg, vib]
model_name = 'reg'
random_seed = 2

#linear regression
reg_epochs = 1000
reg_lr = 2e-1

# min age
# age 0
# height 9
# weight 16
val_list = [0, 9, 16]
sen_list = [2, 3, 4, 5, 6, 7, 13, 14, 15]


# logger
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

    '''
    for target_col in val_list:
        min_val, max_val = np.min(x[:, target_col]), np.max(x[:, target_col]) 
        x[:, target_col] = (x[:, target_col] - min_val) / (max_val - min_val)
        min_val, max_val = np.min(x[:, target_col]), np.max(x[:, target_col])
    '''
    t, target_cols = extract_target(x, target_str, featnames)

    if target_str == "special":
        cnt = np.zeros(8)
        tot = x.shape[0]
        train_list, test_list = [], []
        for i in range(tot):
            for j in range(2, 8):
                if x[i, j] == 1:
                    idx = j
            if cnt[idx] < 300 and idx in [2, 3, 4]:
                cnt[idx] += 1
                train_list.append(i)
            else:
                test_list.append(i)
        train_x, test_x = x[train_list, :], x[test_list, :]
        train_y, test_y = y[train_list], y[test_list]
        train_t, test_t = t[train_list], t[test_list]
    else:
        train_x, test_x, train_y, test_y, train_t, test_t = train_test_split(x, y, t,
                                                        random_state=random_seed, test_size=0.25)


    if model_name == "reg" or model_name == "def" or model_name == "dp":
        logger.info("=> creating model '{}'".format(model_name))
        model = models.MLP(input_dim=x.shape[1]).cuda()
        logger.info(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=reg_lr)
        epochs = reg_epochs
        test_in = torch.from_numpy(trans_norm(test_x)).float().cuda()
        test_label = torch.from_numpy(test_y).float().cuda().view(-1, 1)

        train_in = torch.from_numpy(trans_norm(train_x)).float().cuda()
        train_label = torch.from_numpy(train_y).float().cuda().view(-1, 1)
        tot = train_in.size(0)

        bs = train_label.size(0)
        mask = get_mask()

        best_error = 100
        best_model = None

        for e in range(epochs):
            model.train()
        
            delta = 2*(1+dim)*(1+dim)
            num_phi_1 = dim
            num_phi_2 = dim ** 2
            
            input = train_in[:, :]
            label = train_label[:, :]

            out = model(input)
            loss = torch.mean((label - out) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            filename = os.path.join(save_path, 'model_latest.pth')
            torch.save(
                {
                    'epoch': e,
                    'state_dict': model.cpu().state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': loss
                }, filename)
            model.cuda()

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
                print("Epoch:{}\tTrain Error:{:.4f}\tTest Error:{:.4f}".format(e, train_error, test_error))
                # logger.info('Epoch: [{}/{}]\t'
                #         'Train Error: {:.4f}\t'
                #         'Test Error: {:.4f}.'.format(
                #             e,
                #             reg_epochs,
                #             train_error,
                #             test_error))

        # train_error, test_error = engine(model, model_name, train_x, test_x, 
                                    # train_y, test_y, train_t, target_cols)
    
    else:
        print("Model does not exist")
        exit()
    
    

    if target_str in ["height", "weight", "age"]:
        mae = inver_continuous(model, model_name, train_x, train_y, train_t, target_cols, min_val, max_val)
        print("Model name:{}\tTarget Str:{}\tTrain Error:{:.4f}\tTest Error:{:.4f}\tAttack MAE:{:.2f}".format(
            model_name, target_str, train_error, test_error, mae))
    else:
        attack_acc = inver(model, model_name, train_x, train_y, train_t, target_cols)
        print("Model name:{}\tTarget Str:{}\tTrain Error:{:.4f}\tTest Error:{:.4f}\tAttack Acc:{:.2f}".format(
            model_name, target_str, train_error, test_error, attack_acc * 100))



if __name__ == "__main__":
    main()


