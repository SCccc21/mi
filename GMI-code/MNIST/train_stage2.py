from utils import *
from classify import *
from generator import *
from discri import *
from torch.utils.data import DataLoader
from torch.optim import Adadelta, Adam
from torch.nn import BCELoss, DataParallel
from torchvision.utils import save_image
from torch.autograd import grad
import torchvision.transforms as transforms
import torch
import time
import random
import os, logging
import numpy as np
from attack import inversion
from dist_attack import dist_inversion
from generator import *
from sklearn.model_selection import GridSearchCV


#logger
def get_logger():
	logger_name = "main-logger"
	logger = logging.getLogger(logger_name)
	logger.setLevel(logging.INFO)
	handler = logging.StreamHandler()
	fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
	handler.setFormatter(logging.Formatter(fmt))
	logger.addHandler(handler)
	return logger



if __name__ == "__main__":
	os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
	# os.environ["CUDA_VISIBLE_DEVICES"] = '4, 5, 6, 7'

	global logger
	logger = get_logger()
	dataset_name = "mnist"
	improved_flag = True

	
	print("Using improved GAN:", improved_flag)
   
	z_dim = 100

	path_G = '/home/sichen/models/improvedGAN/improved_mb_mnist_G_entropy.tar'
	path_D = '/home/sichen/models/improvedGAN/improved_mb_mnist_D_entropy.tar'
	
	# path_G = './Attack/attack_models/MNIST_G.tar'
	# path_D = './Attack/attack_models/MNIST_D.tar'

	path_T = '/home/sichen/models/target_model/target_ckp/mnist_cnn_target_99.94.tar'
	path_E = '/home/sichen/models/target_model/target_ckp/mnist_cnn_eval_99.35.tar'

	###########################################
	###########     load model       ##########
	###########################################
	# no mask
	G = GeneratorMNIST(z_dim)
	G = torch.nn.DataParallel(G).cuda()
	if improved_flag == True:
		D = MinibatchDiscriminator_MNIST()
	else:
		D = DGWGAN32()
	
	D = torch.nn.DataParallel(D).cuda()
	ckp_G = torch.load(path_G)
	G.load_state_dict(ckp_G['state_dict'], strict=False)
	ckp_D = torch.load(path_D)
	D.load_state_dict(ckp_D['state_dict'], strict=False)

	T = MCNN(5)
	T = torch.nn.DataParallel(T).cuda()
	ckp_T = torch.load(path_T)
	T.load_state_dict(ckp_T['state_dict'], strict=False)

	E = SCNN(10)
	E = torch.nn.DataParallel(E).cuda()
	ckp_E = torch.load(path_E)
	E.load_state_dict(ckp_E['state_dict'], strict=False)



	############         attack     ###########
	logger.info("=> Begin attacking ...")

	aver_acc, aver_acc5, aver_var = 0, 0, 0
	

	for i in range(5):
		print('-------------------------')
		iden = torch.from_numpy(np.arange(5))
		# acc, acc5, var = inversion(G, D, T, E, iden, itr=i, lr=2e-2, momentum=0.9, lamda=100, iter_times=1500, clip_range=1, improved=improved_flag)
		acc, acc5, var = dist_inversion(G, D, T, E, iden, itr=i, lr=2e-2, momentum=0.9, lamda=100, iter_times=1500, clip_range=1, improved=improved_flag)
		aver_acc += acc / 5
		aver_acc5 += acc5 / 5
		aver_var += var / 5

	print("Average Acc:{:.2f}\tAverage Acc5:{:.2f}\tAverage Acc_var:{:.4f}".format(aver_acc, aver_acc5, aver_var))

