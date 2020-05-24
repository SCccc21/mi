import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
	def __init__(self, input_dim=17):
		super(MLP, self).__init__()
		self.input_dim = input_dim
		self.layer = nn.Linear(self.input_dim, 1, bias=False)

	def forward(self, x):
		return self.layer(x)


class MLP_vib(nn.Module):
	def __init__(self, input_dim=17):
		super(MLP_vib, self).__init__()
		self.input_dim = input_dim
		self.k = self.input_dim // 2
		self.st_layer = nn.Linear(self.input_dim, self.k * 2)
		self.fc_layer = nn.Linear(self.k, 1)


	def forward(self, x):
		statis = self.st_layer(x)
		mu, std = statis[:, :self.k], statis[:, self.k:]
		std = F.softplus(std-5, beta=1)
		eps = torch.FloatTensor(std.size()).normal_().cuda()
		res = mu + std * eps
		out = self.fc_layer(res)
		return out, mu, std

class MLP_sen(nn.Module):
	def __init__(self, input_dim=17):
		super(MLP_sen, self).__init__()
		self.input_dim = input_dim
		self.k = self.input_dim // 2
		self.st_layer = nn.Linear(self.input_dim, self.k * 2)
		self.fc_layer = nn.Linear(self.k, 1)


	def forward(self, x, x_sen):
		statis = self.st_layer(x)
		mu, std = statis[:, :self.k], statis[:, self.k:]
		std = F.softplus(std-5, beta=1)
		eps = torch.FloatTensor(std.size()).normal_().cuda()
		res = mu + std * eps
		out = self.fc_layer(res)

		statis_sen = self.st_layer(x_sen)
		mu_sen, std_sen = statis_sen[:, :self.k], statis_sen[:, self.k:]
		std_sen = F.softplus(std_sen-5, beta=1)
		
		return out, mu_sen, std_sen


