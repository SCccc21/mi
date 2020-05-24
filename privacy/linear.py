from work import load_iwpc, extract_target, inver, inver_continuous, engine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from math import sqrt
import numpy as np
import models, os

# [cyp2c9, vkorc1]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

target_str = "vkorc1"
data_folder = 'data'
# [reg, vib]
model_name = 'def'
random_seed = 2
# min age
# age 0
# height 9
# weight 16
val_list = [0, 9, 16]
sen_list = [2, 3, 4, 5, 6, 7, 13, 14, 15]

# [-1, 1]
def trans_project(y):
    min_y, max_y = np.min(y[:]), np.max(y[:])
    y = (y - min_y) / (max_y - min_y)
    y = (y - 0.5) * 2
    return y

if __name__ == "__main__":
    # [0, 9, 16]
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
        train_x, test_x, train_y, test_y, train_t, test_t = train_test_split(x, y, t, random_state=random_seed, test_size=0.25)

    if model_name == "reg" or model_name == "def" or model_name == "dp":
        model = models.MLP(input_dim=x.shape[1]).cuda()
        train_error, test_error = engine(model, model_name, train_x, test_x, train_y, test_y, train_t, target_cols)
    elif model_name == "vib":
        model = models.MLP_vib(input_dim=x.shape[1]).cuda()
        train_error, test_error = engine(model, model_name, train_x, test_x, train_y, test_y)
    elif model_name == "sen":
        model = models.MLP_sen(input_dim=x.shape[1]).cuda()
        train_error, test_error = engine(model, model_name, train_x, test_x, train_y, test_y)
    elif model_name == "sklearn":
        model = linear_model.Ridge(alpha=0.5)
        model.fit(train_x, train_y)
        train_error = sqrt(mean_squared_error(train_y, model.predict(train_x)))
        test_error = sqrt(mean_squared_error(test_y, model.predict(test_x)))
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


