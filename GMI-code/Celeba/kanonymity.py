import torch, os, time, random, generator, discri, classify, utils
import numpy as np 
import torch.nn as nn
import torchvision.utils as tvls
import torch.nn.functional as F
from utils import log_sum_exp, save_tensor_images

device = "cuda"
num_classes = 1000

# generator, discriminator, target model,
def inversion_grad_constraint_k(G, D, T, E, iden, lr=2e-2, momentum=0.9, lamda=100, lamda2=10, iter_times=1500, clip_range=1, improved=False):
	iden = iden.view(-1).long().cuda()
	criterion = nn.CrossEntropyLoss().cuda()
	bs = iden.shape[0]
	
	G.eval()
	D.eval()
	T.eval()
	E.eval()

	z_hat = torch.zeros(bs, 100)
	flag = torch.zeros(bs)
	least_seed_need = torch.ones(bs).int() * 1000

	for random_seed in range(30):
		tf = time.time()
		r_idx = random_seed
		
		torch.manual_seed(random_seed) 
		torch.cuda.manual_seed(random_seed) 
		np.random.seed(random_seed) 
		random.seed(random_seed)

		z = torch.randn(bs, 100).cuda().float()
		z.requires_grad = True
		v = torch.zeros(bs, 100).cuda().float()
			
		for i in range(iter_times):
			fake = G(z)
			label = D(fake)
			out = T(fake)[-1]
			
			if z.grad is not None:
				z.grad.data.zero_()
			for param in T.parameters():
				if param.grad is not None:
					param.grad.data.zero_()	

			Prior_Loss = - label.mean()
			Iden_Loss = criterion(out, iden)

			Iden_Loss.backward(retain_graph=True)
			Grad_Loss = 0
			for param in T.parameters():
				# print(param.grad.shape)
				if param.grad is None:
					# print("None")
					continue
				Grad_Loss += param.grad.data.mean().abs()
				
			# import pdb; pdb.set_trace()
			Grad_Loss = Grad_Loss.mean().abs()

			Total_Loss = Prior_Loss + lamda * Iden_Loss + lamda2 * Grad_Loss

			if z.grad is not None:
				z.grad.data.zero_()
			Total_Loss.backward()
			
			v_prev = v.clone()
			gradient = z.grad.data
			v = momentum * v - lr * gradient
			z = z + ( - momentum * v_prev + (1 + momentum) * v)
			z = torch.clamp(z.detach(), -clip_range, clip_range).float()
			z.requires_grad = True

			Prior_Loss_val = Prior_Loss.item()
			Iden_Loss_val = Iden_Loss.item()
			Grad_Loss_val = Grad_Loss.item()

			if (i+1) % 300 == 0:
				fake_img = G(z.detach())
				eval_prob = E(utils.low2high(fake_img))[-1]
				eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
				acc = iden.eq(eval_iden.long()).sum().item() * 1.0 / bs
				print("Iteration:{}\tPrior Loss:{:.2f}\tIden Loss:{:.2f}\tGrad Loss:{:.2f}\tAttack Acc:{:.2f}".format(i+1, Prior_Loss_val, Iden_Loss_val, Grad_Loss_val, acc))
			
		fake = G(z)
		score = T(fake)[-1]
		eval_prob = E(utils.low2high(fake))[-1]
		eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
		
		cnt = 0
		for i in range(bs):
			gt = iden[i].item()
 
			if eval_iden[i].item() == gt:
				cnt += 1
				flag[i] = 1
				if r_idx < least_seed_need[i]:
					least_seed_need[i] = r_idx
		
		interval = time.time() - tf
		print("Seed:{}\tTime:{:.2f}\tAcc:{:.2f}\t".format(r_idx, interval, cnt * 1.0 / bs))

		flag_sum = torch.sum(flag) * 1.0 / bs
		if flag_sum == 1:
			return least_seed_need + 1
		
		torch.cuda.empty_cache()

	return least_seed_need + 1



def inversion_k(G, D, T, E, iden, lr=2e-2, momentum=0.9, lamda=100, iter_times=1500, clip_range=1, improved=False):
	iden = iden.view(-1).long().cuda()
	criterion = nn.CrossEntropyLoss().cuda()
	bs = iden.shape[0]
	
	G.eval()
	D.eval()
	T.eval()
	E.eval()

	z_hat = torch.zeros(bs, 100)
	flag = torch.zeros(bs)
	least_seed_need = torch.ones(bs).int() * 1000

	for random_seed in range(30):
		tf = time.time()
		r_idx = random_seed
		
		torch.manual_seed(random_seed) 
		torch.cuda.manual_seed(random_seed) 
		np.random.seed(random_seed) 
		random.seed(random_seed)

		z = torch.randn(bs, 100).cuda().float()
		z.requires_grad = True
		v = torch.zeros(bs, 100).cuda().float()
			
		for i in range(iter_times):
			fake = G(z)
			if improved == True:
				_, label =  D(fake)
			else:
				label = D(fake)
			
			out = T(fake)[-1]
			
			
			if z.grad is not None:
				z.grad.data.zero_()

			if improved:
				Prior_Loss = - torch.mean(F.softplus(log_sum_exp(label)))
			else:
				Prior_Loss = - label.mean()
			Iden_Loss = criterion(out, iden)
			Total_Loss = Prior_Loss + lamda * Iden_Loss

			Total_Loss.backward()
			
			v_prev = v.clone()
			gradient = z.grad.data
			v = momentum * v - lr * gradient
			z = z + ( - momentum * v_prev + (1 + momentum) * v)
			z = torch.clamp(z.detach(), -clip_range, clip_range).float()
			z.requires_grad = True

			Prior_Loss_val = Prior_Loss.item()
			Iden_Loss_val = Iden_Loss.item()

			if (i+1) % 300 == 0:
				fake_img = G(z.detach())
				eval_prob = E(utils.low2high(fake_img))[-1]
				eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
				acc = iden.eq(eval_iden.long()).sum().item() * 1.0 / bs
				print("Iteration:{}\tPrior Loss:{:.2f}\tIden Loss:{:.2f}\tAttack Acc:{:.2f}".format(i+1, Prior_Loss_val, Iden_Loss_val, acc))
			
		fake = G(z)
		score = T(fake)[-1]
		eval_prob = E(utils.low2high(fake))[-1]
		eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
		
		cnt = 0
		for i in range(bs):
			gt = iden[i].item()
			if eval_iden[i].item() == gt:
				cnt += 1
				flag[i] = 1
				if r_idx < least_seed_need[i]:
					least_seed_need[i] = r_idx
		
		interval = time.time() - tf
		print("Seed:{}\tTime:{:.2f}\tAcc:{:.2f}\t".format(r_idx, interval, cnt * 1.0 / bs))

		flag_sum = torch.sum(flag) * 1.0 / bs
		if flag_sum == 1:
			return least_seed_need + 1

		torch.cuda.empty_cache()

	return least_seed_need + 1

if __name__ == '__main__':
	pass



	
	
	
	
	

	
	
		

	

