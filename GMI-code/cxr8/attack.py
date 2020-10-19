import torch, os, time, random, generator, discri, classify, utils
import numpy as np 
import torch.nn as nn
import torchvision.utils as tvls
import torch.nn.functional as F
from utils import log_sum_exp, save_tensor_images
import statistics 

device = "cuda"
num_classes = 7
save_img_dir = './attack_imgs_cxr_origin'
os.makedirs(save_img_dir, exist_ok=True)


def inversion(G, D, T, E, iden, itr, lr=2e-2, momentum=0.9, lamda=1000, iter_times=1500, clip_range=1, improved=False):
	iden = iden.view(-1).long().cuda()
	criterion = nn.CrossEntropyLoss().cuda()
	bs = iden.shape[0]
	
	G.eval()
	D.eval()
	T.eval()
	E.eval()

	max_score = torch.zeros(bs)
	max_iden = torch.zeros(bs)
	max_prob = torch.zeros(bs, 7)
	z_hat = torch.zeros(bs, 100)
	flag = torch.zeros(bs)
	no = torch.zeros(bs) # index for saving all success attack images

	res = []
	res5 = []
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
			
			out = T(fake)
			
			
			if z.grad is not None:
				z.grad.data.zero_()

			if improved:
				Prior_Loss = torch.mean(F.softplus(log_sum_exp(label))) - torch.mean(log_sum_exp(label))
				# Prior_Loss =  torch.mean(F.softplus(log_sum_exp(label))) - torch.mean(label.gather(1, iden.view(-1, 1)))  #1 class prior
			else:
				Prior_Loss = - label.mean()
			Iden_Loss = criterion(out, iden)
			# import pdb; pdb.set_trace()

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
				eval_prob = E(fake_img)
				eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
				acc = iden.eq(eval_iden.long()).sum().item() * 1.0 / bs
				print("Iteration:{}\tPrior Loss:{:.2f}\tIden Loss:{:.2f}\tAttack Acc:{:.2f}".format(i+1, Prior_Loss_val, Iden_Loss_val, acc))
			
		fake = G(z)
		score = T(fake)
		eval_prob = E(fake)
		eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
		# save_tensor_images(fake.detach(), os.path.join(save_img_dir, "attack_result_image_{}_{}.png".format(iden[0], r_idx)), nrow = 10)
		
		cnt, cnt5 = 0, 0
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
				save_tensor_images(best_img.detach(), os.path.join(save_img_dir, "{}_attack_iden_{}_{}.png".format(itr, i, int(no[i]))))
				no[i] += 1
			_, top5_idx = torch.topk(eval_prob[i], 5)
			if gt in top5_idx:
				cnt5 += 1
				
		
		interval = time.time() - tf
		print("Time:{:.2f}\tAcc:{:.2f}\t".format(interval, cnt * 1.0 / bs))
		res.append(cnt * 1.0 / bs)
		res5.append(cnt5 * 1.0 / bs)
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
	# acc, acc_5, acc_5_prev = correct * 1.0 / bs, cnt5 * 1.0 / bs, correct_5 * 1.0 / bs
	acc, acc_5 = statistics.mean(res), statistics.mean(res5)
	acc_var = statistics.variance(res)
	acc_var5 = statistics.variance(res5)
	print("Acc:{:.2f}\tAcc_5:{:.2f}\tAcc_var:{:.4f}\tAcc_var5:{:.4f}".format(acc, acc_5, acc_var, acc_var5))
	
	return acc, acc_5, acc_var, acc_var5

if __name__ == "__main__":
	pass
	