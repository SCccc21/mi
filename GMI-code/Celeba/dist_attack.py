import torch, os, time, random, generator, discri, classify, utils
import numpy as np 
import torch.nn as nn
import torchvision.utils as tvls
import torch.nn.functional as F
from utils import log_sum_exp, save_tensor_images
from torch.autograd import Variable
import torch.optim as optim
import torch.autograd as autograd
import statistics 

device = "cuda"
num_classes = 1000
save_img_dir = '/home/sichen/mi/GMI-code/Celeba/fid/fid_dist_ffhq_mb_entropy'
os.makedirs(save_img_dir, exist_ok=True)

def reparameterize(mu, logvar):
	"""
	Reparameterization trick to sample from N(mu, var) from
	N(0,1).
	:param mu: (Tensor) Mean of the latent Gaussian [B x D]
	:param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
	:return: (Tensor) [B x D]
	"""
	std = torch.exp(0.5 * logvar)
	eps = torch.randn_like(std)
	return eps * std + mu


def dist_inversion(G, D, T, E, iden, itr, lr=2e-2, momentum=0.9, lamda=100, iter_times=1500, clip_range=1, improved=False, num_seeds=5):
	iden = iden.view(-1).long().cuda()
	criterion = nn.CrossEntropyLoss().cuda()
	bs = iden.shape[0]
	
	G.eval()
	D.eval()
	T.eval()
	E.eval()

	# max_score = torch.zeros(bs)
	# max_iden = torch.zeros(bs)
	# max_prob = torch.zeros(bs, num_classes)
	# z_hat = torch.zeros(bs, 100)
	# flag = torch.zeros(bs)
	no = torch.zeros(bs) # index for saving all success attack images

	tf = time.time()

	#NOTE
	mu = Variable(torch.zeros(bs, 100), requires_grad=True)
	log_var = Variable(torch.zeros(bs, 100), requires_grad=True)
	
	params = [mu, log_var]
	solver = optim.Adam(params, lr=lr)
		
	for i in range(iter_times):
		z = reparameterize(mu, log_var)
		fake = G(z)
		if improved == True:
			_, label =  D(fake)
		else:
			label = D(fake)
		
		out = T(fake)[-1]
		
		for p in params:
			if p.grad is not None:
				p.grad.data.zero_()

		if improved:
			Prior_Loss = torch.mean(F.softplus(log_sum_exp(label))) - torch.mean(log_sum_exp(label))
			# Prior_Loss =  torch.mean(F.softplus(log_sum_exp(label))) - torch.mean(label.gather(1, iden.view(-1, 1)))  #1 class prior
		else:
			Prior_Loss = - label.mean()
		Iden_Loss = criterion(out, iden)

		Total_Loss = Prior_Loss + lamda * Iden_Loss
		# import pdb; pdb.set_trace()

		Total_Loss.backward()
		solver.step()
		
		z = torch.clamp(z.detach(), -clip_range, clip_range).float()

		Prior_Loss_val = Prior_Loss.item()
		Iden_Loss_val = Iden_Loss.item()

		if (i+1) % 300 == 0:
			fake_img = G(z.detach())
			eval_prob = E(utils.low2high(fake_img))[-1]
			eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
			acc = iden.eq(eval_iden.long()).sum().item() * 1.0 / bs
			print("Iteration:{}\tPrior Loss:{:.2f}\tIden Loss:{:.2f}\tAttack Acc:{:.2f}".format(i+1, Prior_Loss_val, Iden_Loss_val, acc))
			
	interval = time.time() - tf
	print("Time:{:.2f}".format(interval))
	
	res = []
	res5 = []
	for random_seed in range(num_seeds):
		tf = time.time()
		z = reparameterize(mu, log_var)
		fake = G(z)
		score = T(fake)[-1]
		eval_prob = E(utils.low2high(fake))[-1]
		eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
		
		cnt, cnt5 = 0, 0
		for i in range(bs):
			gt = iden[i].item()
			'''
			if score[i, gt].item() > max_score[i].item():
				max_score[i] = score[i, gt]
				max_iden[i] = eval_iden[i]
				max_prob[i] = eval_prob[i]
				# z_hat[i, :] = z[i, :]
			'''
			if eval_iden[i].item() == gt:
				cnt += 1
				# flag[i] = 1
				# best_img = G(z)[i]
				# save_tensor_images(best_img.detach(), os.path.join(save_img_dir, "{}_attack_iden_{}_{}.png".format(itr, iden[0]+i+1, int(no[i]))))
				no[i] += 1
			_, top5_idx = torch.topk(eval_prob[i], 5)
			if gt in top5_idx:
				cnt5 += 1
				
		interval = time.time() - tf
		print("Time:{:.2f}\tSeed:{}\tAcc:{:.2f}\t".format(interval, random_seed, cnt * 1.0 / bs))
		res.append(cnt * 1.0 / bs)
		res5.append(cnt5 * 1.0 / bs)

		torch.cuda.empty_cache()

	'''
	correct = 0
	cnt5 = 0
	for i in range(bs):
		gt = iden[i].item()
		if max_iden[i].item() == gt:
			correct += 1
			
		# top5
		_, top5_idx = torch.topk(max_prob[i], 5)
		if gt in top5_idx:
			cnt5 += 1
	'''
		
	
	acc, acc_5 = statistics.mean(res), statistics.mean(res5)
	acc_var = statistics.variance(res)
	print("Acc:{:.2f}\tAcc_5:{:.2f}\tAcc_var:{:.4f}".format(acc, acc_5, acc_var))
	
	return acc, acc_5, acc_var
