from work import load_iwpc, extract_target, inver, engine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from math import sqrt
import pandas as pd
import numpy as np
import models
import json

class MyDecisionTreeRegressor():
    def __init__(self, max_depth=5, min_samples_split=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
    
    def convert_to_df(self, arr):
        return pd.DataFrame(arr)

    def get_sse_class(self, y, classes):
        min_sse = None
        for classv in classes:
            sse = ((y - classv)**2).sum().tolist()[0]
            if not min_sse or sse < min_sse:
                min_sse = sse
        return min_sse

    def get_mean_sse(self, y):
        mean = y.mean().tolist()[0]
        return ((y - mean)**2).sum().tolist()[0]

    def select_optimal_splitting(self, X_df, y_df):
        n_rows, feat_len = X_df.shape
        #X_df, y_df = self.convert_to_df(X), self.convert_to_df(y)
        #X_df.columns = [str(i)  for i in range(feat_len)]
        min_sse = None
        #sse_details = (Feature, value)
        sse_details = None
        for i in range(feat_len):
            curr_col = X_df[str(i)]
            uniq_values = np.sort(curr_col.unique())
            for value in uniq_values[:-1]:
                indx_1, indx_2 = curr_col <= value, curr_col > value
                y_1 = y_df[indx_1].dropna()
                y_2 = y_df[indx_2].dropna()
                sse_y1, sse_y2 = self.get_mean_sse(y_1), self.get_mean_sse(y_2)
                
                sse = sse_y1 + sse_y2
                if min_sse == None or sse < min_sse:
                    min_sse = sse
                    sse_details = (str(i), value)
        return {
                'idx': sse_details[0],
                'threshold': sse_details[1],
        }

    def calculate_c(self, y):
        n_rows = y.shape[0]
        y = y.values.flatten()
        mean = y.sum() / n_rows
        var = ((y - mean) ** 2).mean()
        
        return (mean, var)

    def fit(self, X, y):
        '''
        Inputs:
        X: Train feature data, type: numpy array, shape: (N, num_feature)
        Y: Train label data, type: numpy array, shape: (N,)
        You should update the self.root in this function.
        '''
        n_rows, feat_len = X.shape
        X_df, y_df = self.convert_to_df(X), self.convert_to_df(y)
        X_df.columns = [str(i)  for i in range(feat_len)]
        self.root = self.fit_aux(X_df, y_df, 0)
        #print(self.root)

    def fit_aux(self, X, y, height):
        '''
        Inputs:
        X: Train feature data, type: numpy array, shape: (N, num_feature)
        Y: Train label data, type: numpy array, shape: (N,)
        You should update the self.root in this function.
        '''
        curr_depth = 0
        #Create a function to select the optimal splitting given the dataset.
        split_info = self.select_optimal_splitting(X, y)
        #Update self.root
        resp = {
            'splitting_variable': int(split_info['idx']),
            'splitting_threshold': split_info['threshold']
        }
        

        left_split, y_left = X[X[split_info['idx']] <= split_info['threshold']].dropna(), y[X[split_info['idx']] <= split_info['threshold']].dropna()
        right_split, y_right = X[X[split_info['idx']] > split_info['threshold']].dropna(), y[X[split_info['idx']] > split_info['threshold']].dropna()
        
        #print(left_split, y_left)
        #print("-----------------------------------------------------------------------")
        #print(right_split, y_right)
        #print("--------------------------------------------------------------------")
        #print("********************************************************************")
        if left_split.shape[0] < self.min_samples_split or height + 1 == self.max_depth:
            #It is a leaf node
            resp.update({
                'left': self.calculate_c(y_left)
            })
        else:
            #Do a Recursive iteration
            resp.update({
                'left': self.fit_aux(left_split, y_left, height + 1)
            })

        if right_split.shape[0] < self.min_samples_split or height + 1 == self.max_depth:
            #It is a leaf node
            resp.update({
                'right': self.calculate_c(y_right)
            })
        else:
            #Do a Recursive iteration.
            resp.update({
                'right': self.fit_aux(right_split, y_right, height + 1)
            })
        return resp

    def pred_row(self, row):
        curr_node = self.root
        while type(curr_node) == dict:
            idx = curr_node['splitting_variable']
            val = row[idx]
            if val <= curr_node['splitting_threshold']:
                curr_node = curr_node["left"]
            else:
                curr_node = curr_node["right"]
        mu, std = curr_node
        eps = np.random.randn()
        res = mu + std * eps
        return mu

    def predict(self, X):
        '''
        :param X: Feature data, type: numpy array, shape: (N, num_feature)
        :return: y_pred: Predicted label, type: numpy array, shape: (N,)
        '''
        y_pred = []
        for row in X:
            y_pred.append(self.pred_row(row))
        return np.array(y_pred)

    def get_model_string(self):
        model_dict = self.root
        return model_dict

    def save_model_to_json(self, file_name):
        model_dict = self.root
        with open(file_name, 'w') as fp:
            #splitting a nodeplitting a node into two child nod into two child nod
            json.dump(model_dict, fp)

    def load_model(self, file_name):
        with open(file_name) as json_file:
            self.root = json.load(json_file)

# [cyp2c9, vkorc1]
target_str = "cyp2c9"
data_folder = 'data'
model_name = "tree"

depth = 13
split = 5
file_name = "Tree_depth_" + str(depth) + "_split_" + str(split) + '.json'


if __name__ == "__main__":
    x, y, featnames = load_iwpc(data_folder)
    
    model = MyDecisionTreeRegressor(max_depth=depth, min_samples_split=split)
    t, target_cols, dist = extract_target(x, target_str, featnames)
    
    train_x, test_x, train_y, test_y, train_t, test_t = train_test_split(x, y, t, test_size=0.25)
    model.fit(train_x, train_y)
    #model.save_model_to_json(file_name)
    #model.load_model(file_name)

    train_error = sqrt(mean_squared_error(train_y, model.predict(train_x)))
    test_error = sqrt(mean_squared_error(test_y, model.predict(test_x)))
    inv_accuracy = inver(model, model_name, train_x, train_y, train_t, target_cols)
    print("Train Error:{:.2f}\tTest Error:{:.2f}\tinv_accuracy:{:.2f}".format(train_error, test_error, inv_accuracy))

    '''
    for depth in range(5, 25, 2):

        model = DecisionTreeRegressor(max_depth=depth, min_samples_split = 5)
        r_emp = get_empirical_error(model, x, y) #root mean squared
        r_cv = get_cross_validation_error(model, x, y) #root mean squared


        train_X, test_X, train_y, test_y, train_t, test_t = train_test_split(x, y, t, test_size=0.25)
        model.fit(train_X, train_y)
        train_correct = inversion(model, dist, train_X, train_y, train_t, target_cols, r_emp)
        #test_correct = inversion(model, dist, test_X, test_y, test_t, target_cols, r_cv)
        print("Train Error:{:.2f}\tTest Error:{:.2f}\tAttack Accuracy:{:.2f}".format(r_emp, r_cv, train_correct))
    '''
    

