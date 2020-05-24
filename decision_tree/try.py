import numpy as np
import pandas as pd

def tree_invert(model, X, y, target_col):
	assert X.shape[0] == y.shape[0] #check that X and y have compatible dimensions

	guesses = []

	for i in range(X.shape[0]): #iterate over the rows of X and y
		row_X = np.stack([X[i] for _ in range(num_variants)]) #create copies of X[i]
		row_X[0, target_cols] = 0
		row_X[1, target_cols] = 1
		
		row_y = np.repeat(y[i], num_variants)

		errors = row_y - model.predict(row_X)
		scores = np.abs(errors)
		guesses.append(np.argmin(scores))
	    
	return np.array(guesses)
