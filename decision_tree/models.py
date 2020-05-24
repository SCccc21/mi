import pandas as pd
import numpy as np
import json, random
from math import log

random_seed = 2
np.random.seed(random_seed) #numpy
random.seed(random_seed)

target_col = '5'

# mode in ["mutual", "priority"]

class DecisionTree():
    def __init__(self, max_depth=5, min_samples_split=5, mode="mutual", args=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.mode = mode
        self.args = args
        self.root = None
    
    def convert_to_df(self, arr):
        return pd.DataFrame(arr)

    def get_gini(self, y):
        if y.shape[0] == 0:
            return 1
        y = y.values.flatten()
        tot = y.shape[0]
        num_1 = np.sum(y)
        num_0 = tot - num_1
        p_0 = num_0 * 1.0 / tot
        p_1 = num_1 * 1.0 / tot
        gini = 1 - p_0 ** 2 - p_1 ** 2
        entropy = p_0 * log(p_0+1e-7) + p_1 * log(p_1+1e-7)
        '''
        if num_0 > num_1:
            entropy = p_0 * log(p_0+1e-7)
        else:
            entropy = p_1 * log(p_1+1e-7)
        '''
        if self.mode == "mutual":
            return gini + self.args * entropy
        else:
            return gini

    def select_optimal_splitting(self, height, X_df, y_df):
        n_rows, feat_len = X_df.shape
        min_gini = None
        #sse_details = (Feature, value)
        sse_details = None
        for i in range(feat_len):
            if self.mode == "priority" and str(i) == target_col and height < self.args:
                continue
            curr_col = X_df[str(i)]
            idx_1, idx_2 = curr_col == 0, curr_col > 0

            y_1 = y_df[idx_1]
            y_2 = y_df[idx_2]
            gini_y1, gini_y2 = self.get_gini(y_1), self.get_gini(y_2)
            gini = (y_1.shape[0] / n_rows) * gini_y1 + (y_2.shape[0] / n_rows) * gini_y2

            if min_gini == None or gini < min_gini:
                min_gini = gini
                sse_details = (str(i), [0])
        
        return {
                'idx': sse_details[0],
                'list': sse_details[1],
        }
           

    def calculate_c(self, y):
        y = y.values.flatten()
        tot = y.shape[0]
        num_1 = np.sum(y)
        if num_1 > tot / 2:
            return 1
        else:
            return 0
        
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
        assert X.shape[0] >= self.min_samples_split
        '''
        Inputs:
        X: Train feature data, type: pd dataframe, shape: (N, num_feature)
        Y: Train label data, type: pd dataframe, shape: (N,)
        You should update the self.root in this function.
        '''
        curr_depth = 0
        #Create a function to select the optimal splitting given the dataset.
        split_info = self.select_optimal_splitting(height, X, y)
        #Update self.root
        resp = {
            'split_variable': int(split_info['idx']),
            'split_list': split_info['list']
        }
        
        #print(X[split_info['idx']])
        #left_split = X[split_info['idx']] in split_info['list']
        n_rows = X.shape[0]
        curr_col = int(split_info['idx'])
        left_list, right_list = [], []

        for i in range(n_rows):
            if X.iloc[i, curr_col] in split_info['list']:
                left_list.append(i)
            else:
                right_list.append(i)

        left_split, y_left = X.iloc[left_list], y.iloc[left_list]
        right_split, y_right = X.iloc[right_list], y.iloc[right_list]

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
            idx = curr_node['split_variable']
            val = row[idx]
            if val in curr_node['split_list']:
                curr_node = curr_node["left"]
            else:
                curr_node = curr_node["right"]
       
        return curr_node

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

