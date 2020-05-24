import os
import csv, random
import numpy as np 
import pandas as pd
from models import DecisionTree
from sklearn import tree
from sklearn.model_selection import train_test_split

file_name = "GSShappiness.csv"

random_seed = 2
np.random.seed(random_seed) #numpy
random.seed(random_seed)

def load_data(data_file):
	X, data, label = [], [], []
	except_field = ['id']
	with open(data_file) as csvfile:
		reader = csv.DictReader(csvfile)
		col_values = {}
		col_name = []
		label_name = 'watched x-rated movies in the last year'
		for name in reader.fieldnames:
			col_values[name] = []
			if name not in except_field:
				col_name.append(name)

		row_flag = {}
		idx = 0

		for row in reader:
			flag = 1
			for name in col_name:
				if row[name] == '' or row[name] =='OTHER':
					flag = 0
					break
			if flag:
				X.append(row)

		for row in X:
			for name in col_name:
				if row[name] not in col_values[name]:
					col_values[name].append(row[name])

		for name in col_name:
			col_values[name] = sorted(col_values[name])
			print(name, len(col_values[name]))

		for row in X:
			rec = []
			for name in col_name:
				idx = col_values[name].index(row[name])
				if name == label_name:
					label.append(idx)
				else:
					rec.append(idx)
			data.append(rec)
		
	data, label = np.array(data), np.array(label)

	
	binary_list = []
	for c in range(data.shape[1]):
		mx = 0
		for i in range(data.shape[0]):
			mx = max(mx, data[i, c])

		if mx == 1:
			binary_list.append(c)

	data = data[:, binary_list]
	

	print(data.shape, label.shape)

	
	return data, label

def tree_invert(model, X, y, target_col):
	assert X.shape[0] == y.shape[0] #check that X and y have compatible dimensions

	guesses = []
	num_variants = 2
	gt = X[:, target_col]

	for i in range(X.shape[0]): #iterate over the rows of X and y
		row_X = np.stack([X[i] for _ in range(num_variants)]) #create copies of X[i]
		row_X[0, target_col] = 0
		row_X[1, target_col] = 1
		
		row_y = np.repeat(y[i], num_variants)

		errors = row_y - model.predict(row_X)
		scores = np.abs(errors)

		if scores[1] < scores[0]:
			guesses.append(1)
		else:
			guesses.append(0)
		
	guesses = np.array(guesses)
	#print(gt, guesses)

	acc = np.sum(gt == guesses) * 1.0 / X.shape[0]
	return acc

target_col = 5

if __name__ == "__main__":
	x, y = load_data(file_name)
	train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25, random_state=4)

	#clf = DecisionTree(max_depth=5, min_samples_split=5, mode="priority", args=4)
	clf = tree.DecisionTreeClassifier(max_depth=5, min_samples_split=5)
	
	clf.fit(train_x, train_y)
	y_hat = clf.predict(test_x)
	test_acc = np.sum(y_hat == test_y) * 1.0 / test_y.shape[0]
	attack_acc = tree_invert(clf, train_x, train_y, target_col)

	print("Test Acc:{:.2f}\tAttack Acc:{:.2f}".format(test_acc, attack_acc))
