import torch, os, time, random, generator, discri, classify, utils
import numpy as np 
import torch.nn as nn
import torchvision.utils as tvls
import torch.nn.functional as F
from utils import log_sum_exp, save_tensor_images

device = "cuda"
num_classes = 1000
save_img_dir = '/home/sichen/mi/GMI-code/Celeba/fid/fid_combine'
os.makedirs(save_img_dir, exist_ok=True)


def inversion(G, D, T, E, iden, itr, lr=2e-2, momentum=0.9, lamda=100, iter_times=1500, clip_range=1, improved=False):
	iden = iden.view(-1).long().cuda()
	criterion = nn.CrossEntropyLoss().cuda()
	bs = iden.shape[0]
	
	G.eval()
	D.eval()
	T.eval()
	E.eval()

	max_score = torch.zeros(bs)
	max_iden = torch.zeros(bs)
	max_prob = torch.zeros(bs, num_classes)
	z_hat = torch.zeros(bs, 100)
	flag = torch.zeros(bs)
	no = torch.zeros(bs) # index for saving all success attack images

	for random_seed in range(5):
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
				# Prior_Loss = - torch.mean(F.softplus(log_sum_exp(label)))
				# Prior_Loss = torch.mean(F.softplus(log_sum_exp(label))) - torch.mean(log_sum_exp(label))
				Prior_Loss =  torch.mean(F.softplus(log_sum_exp(label))) - torch.mean(label.gather(1, iden.view(-1, 1)))  #1 class prior
			else:
				Prior_Loss = - label.mean()
			Iden_Loss = criterion(out, iden)

			Total_Loss = Prior_Loss + lamda * Iden_Loss
			# import pdb; pdb.set_trace()

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
		# save_tensor_images(fake.detach(), os.path.join(save_img_dir, "attack_result_image_{}_{}.png".format(iden[0], r_idx)), nrow = 10)
		
		cnt = 0
		for i in range(bs):
			gt = iden[i].item()
			if score[i, gt].item() > max_score[i].item():
				max_score[i] = score[i, gt]
				max_iden[i] = eval_iden[i]
				max_prob[i] = eval_prob[i]
				z_hat[i, :] = z[i, :]
			if eval_iden[i].item() == gt:
				cnt += 1
				flag[i] = 1
				best_img = G(z)[i]
				save_tensor_images(best_img.detach(), os.path.join(save_img_dir, "{}_attack_iden_{}_{}.png".format(itr, iden[0]+i+1, int(no[i]))))
				no[i] += 1
				
		
		interval = time.time() - tf
		print("Time:{:.2f}\tAcc:{:.2f}\t".format(interval, cnt * 1.0 / bs))

		torch.cuda.empty_cache()

	correct = 0
	cnt5 = 0
	for i in range(bs):
		gt = iden[i].item()
		if max_iden[i].item() == gt:
			correct += 1
			# best_img = G(z_hat)[i]
			# save_tensor_images(best_img.detach(), os.path.join(save_img_dir, "attack_iden_{}.png".format(iden[0]+i+1)))
			
		# top5
		_, top5_idx = torch.topk(max_prob[i], 5)
		if gt in top5_idx:
			cnt5 += 1
		
	
	correct_5 = torch.sum(flag)
	acc, acc_5, acc_5_prev = correct * 1.0 / bs, cnt5 * 1.0 / bs, correct_5 * 1.0 / bs
	print("Acc:{:.2f}\tAcc_5:{:.2f}\tAcc5_prev:{:.2f}".format(acc, acc_5, acc_5_prev))
	# return acc, acc_5, Prior_Loss_val, Iden_Loss_val
	return acc, acc_5


def natural_grad(G, D, T, E, iden, lr=2e-2, momentum=0.9, lamda=100, lamda2=10, iter_times=1500, clip_range=1, improved=False):
	iden = iden.view(-1).long().cuda()
	criterion = nn.CrossEntropyLoss().cuda()
	bs = iden.shape[0]
	
	G.eval()
	D.eval()
	T.eval()
	E.eval()

	max_score = torch.zeros(bs)
	max_iden = torch.zeros(bs)
	max_prob = torch.zeros(bs, num_classes)
	z_hat = torch.zeros(bs, 100)
	flag = torch.zeros(bs)

	for random_seed in range(5):
		tf = time.time()
		r_idx = random_seed
		
		torch.manual_seed(random_seed) 
		torch.cuda.manual_seed(random_seed) 
		np.random.seed(random_seed) 
		random.seed(random_seed)

		z = torch.randn(bs, 100).cuda().float()
		z.requires_grad = True
		v = torch.zeros(bs, 100).cuda().float()

		lam = 0.01

			
		for i in range(iter_times):
			fake = G(z)
			label = D(fake)
			out = T(fake)[-1]
			
			if z.grad is not None:
				z.grad.data.zero_()
			for name, param in T.named_parameters():
				if param.grad is not None:
					param.grad.data.zero_()	

			Prior_Loss = - label.mean()
			Iden_Loss = criterion(out, iden)

			Iden_Loss.backward(retain_graph=True)
			Grad_Loss = 0

			for name, param in T.named_parameters():
				if param.grad is not None:
					# print(param.grad.shape)
					p_grad_mat = param.grad.data.view(param.grad.data.shape[0], -1) # gradient in matrix form: n_filters * (in_c * kw * kh)
					fim = p_grad_mat.mm(p_grad_mat.t()) / bs
					fim.detach_()
					Ipp = torch.eye(fim.shape[0]).float().cuda()
					# import pdb; pdb.set_trace()
					grad_natural = torch.inverse(fim + lam * Ipp).mm(p_grad_mat)
					Grad_Loss += grad_natural.mean().abs()


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
		# save_tensor_images(fake.detach(), os.path.join(save_img_dir, "attack_result_image_{}_{}_0715.png".format(iden[0], r_idx)), nrow = 10)
		
		cnt = 0
		for i in range(bs):
			gt = iden[i].item()
			if score[i, gt].item() > max_score[i].item():
				max_score[i] = score[i, gt]
				max_iden[i] = eval_iden[i]
				max_prob[i] = eval_prob[i]
				z_hat[i, :] = z[i, :]
 
			if eval_iden[i].item() == gt:
				cnt += 1
				flag[i] = 1
		
		interval = time.time() - tf
		print("Time:{:.2f}\tAcc:{:.2f}\t".format(interval, cnt * 1.0 / bs))

		torch.cuda.empty_cache()

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
	
	correct_5 = torch.sum(flag)
	acc, acc_5, acc_5_prev = correct * 1.0 / bs, cnt5 * 1.0 / bs, correct_5 * 1.0 / bs
	print("Acc:{:.2f}\tAcc_5:{:.2f}\tAcc5_prev:{:.2f}".format(acc, acc_5, acc_5_prev))
	return acc, acc_5

# generator, discriminator, target model,
def inversion_grad_constraint(G, D, T, E, iden, lr=2e-2, momentum=0.9, lamda=100, lamda2=10, iter_times=1500, clip_range=1, improved=False):
	iden = iden.view(-1).long().cuda()
	criterion = nn.CrossEntropyLoss().cuda()
	bs = iden.shape[0]
	
	G.eval()
	D.eval()
	T.eval()
	E.eval()

	max_score = torch.zeros(bs)
	max_iden = torch.zeros(bs)
	max_prob = torch.zeros(bs, num_classes)
	z_hat = torch.zeros(bs, 100)
	flag = torch.zeros(bs)

	for random_seed in range(5):
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
			for param in T.parameters():
				if param.grad is not None:
					param.grad.data.zero_()	

			if improved:
				Prior_Loss = - torch.mean(F.softplus(log_sum_exp(label)))
			else:
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
		# save_tensor_images(fake.detach(), os.path.join(save_img_dir, "attack_result_image_{}_{}.png".format(iden[0], r_idx)), nrow = 10)
		
		cnt = 0
		for i in range(bs):
			gt = iden[i].item()
			if score[i, gt].item() > max_score[i].item():
				max_score[i] = score[i, gt]
				max_iden[i] = eval_iden[i]
				max_prob[i] = eval_prob[i]
				z_hat[i, :] = z[i, :]
 
			if eval_iden[i].item() == gt:
				cnt += 1
				flag[i] = 1
		
		interval = time.time() - tf
		print("Time:{:.2f}\tAcc:{:.2f}\t".format(interval, cnt * 1.0 / bs))

		torch.cuda.empty_cache()

	correct = 0
	cnt5 = 0
	for i in range(bs):
		gt = iden[i].item()
		if max_iden[i].item() == gt:
			correct += 1
			best_img = G(z_hat)[i]
			save_tensor_images(best_img.detach(), os.path.join(save_img_dir, "attack_iden_{}_gc_improved.png".format(iden[0]+i+1)))
		# top5
		_, top5_idx = torch.topk(max_prob[i], 5)
		if gt in top5_idx:
			cnt5 += 1
	
	correct_5 = torch.sum(flag)
	acc, acc_5, acc_5_prev = correct * 1.0 / bs, cnt5 * 1.0 / bs, correct_5 * 1.0 / bs
	print("Acc:{:.2f}\tAcc_5:{:.2f}\tAcc5_prev:{:.2f}".format(acc, acc_5, acc_5_prev))
	return acc, acc_5

if __name__ == '__main__':
	pass



	
	
	
	
	

	
	
		

	

