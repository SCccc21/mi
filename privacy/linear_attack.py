import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# [-1,1]
def trans_project(t):
    min_t, max_t = np.min(t[:]), np.max(t[:])
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
    for i in range(x[0]):
        if t[i] >= 1.5:
            t[i] = 2
        elif t[i] >= 0.5 and t[i]<1.5:
            t[i] = 1
        else:
            t[i] = 0

        for j in range(len(target_cols)):
            x[i, target_cols[j]] = 1 if j == t[i] else 0

        return x
            

class NAT(nn.Module):
	def __init__(self, input_dim=17):
		super(NAT, self).__init__()
		self.input_dim = input_dim
		# self.net = nn.Linear(self.input_dim, 1, bias=False)
        # self.criterion = nn.MSELoss()

    def nat(self, x, y, t, target_cols, eps=0.4, t_val_min=-1, t_val_max=1, iteration):
        t = trans_project(t)
        t_adv = Variable(t.data, requires_grad=True)
        x_adv = merge_t(t_adv.data, x, target_cols)
        for itr in range(iteration):
            tmp1 = np.linalg.inv(np.dot(x_adv.T, x_adv))
            tmp2 = np.dot(x_adv.T, y)
            c_bar = tmp1.dot(tmp2)
            h_adv = np.dot(c_bar.T, x_adv)
            cost = torch.mean((y - h_adv) ** 2)
            true_cost = torch.mean()

            loss = (true_cost - cost).abs().mean()
            self.net.zero_grad()
            if t_adv.grad is not None:
                t_adv.grad.data.fill_(0)
            loss.backward()

            t_adv.grad.sign_()
            t_adv = t_adv - eps*t_adv.grad
            t_adv = where(t_adv > t+eps, t+eps, t_adv)
            t_adv = where(t_adv < t-eps, t-eps, t_adv)
            t_adv = torch.clamp(t_adv, t_val_min, t_val_max)
            x_adv = merge_t(t_adv.data, x_adv, target_cols)

        return x_adv

