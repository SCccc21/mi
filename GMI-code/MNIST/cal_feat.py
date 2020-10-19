from utils import *
from classify import *
from generator import *
from discri import *
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.autograd import grad
import torchvision.transforms as transforms
import torch
import os
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF


def save_knn(I):
	feat_path = './feat/feat.npy'
	info_path = "./feat/info.npy"
	img_folder = './private_domain/'

	listOfFile = os.listdir(img_folder)

	img_list = []
	label_list = []

	cnt = torch.zeros(5)
	for entry in listOfFile:
		_, _, label = os.path.splitext(entry)[0].strip().split('_')
		# if cnt[int(label)-1] > 19:
		# 	continue
		if int(label)-1 in [0,1,2,3,4]:
			img = Image.open(img_folder+entry)
			img = img.convert('L')
			img = TF.to_tensor(img)
			img_list.append(img)
			label_list.append(int(label)-1)
			cnt[int(label)-1] += 1
		
	image = torch.stack(img_list, dim=0)
	# import pdb; pdb.set_trace()
	info = torch.LongTensor(label_list)
	feat, _ = I(image)
	
	np.save(feat_path, feat.detach().cpu().numpy())
	np.save(info_path, info.cpu().numpy())
	print("Success!")


def get_knn_dist(base_path, I):
	listOfFile = os.listdir(base_path)

	img_list = []
	label_list = []
	for entry in listOfFile:
		idx, _, _, label, seed = os.path.splitext(entry)[0].strip().split('_')
		if int(label) in [0,1,2,3,4]: # old image is 1-5, new is 0-4 which replace old one, shouldn't use 5
			img = Image.open(base_path+entry)
			img = img.convert('L')
			img = TF.to_tensor(img)
			img_list.append(img)
			label_list.append(int(label)) #0-4
	
	image = torch.stack(img_list, dim=0)
	print(image.shape)
	iden = torch.LongTensor(label_list)
	feat, _ = I(image)
	dist = calc_knn(feat.detach(), iden, path='./feat')

	return dist, feat.detach()

def psnr(img1, img2):
	mse = torch.mean((img1 - img2) ** 2)
	return 20 * torch.log10(255.0 / torch.sqrt(mse))

if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6,7'

	I = SCNN(10)
	ckp = torch.load('/home/sichen/models/target_model/target_ckp/mnist_cnn_eval_99.35.tar')
	I = torch.nn.DataParallel(I).cuda()
	I.load_state_dict(ckp['state_dict'], strict=False)
	
	print("Loading Backbone Checkpoint ")

	save_knn(I)

	path = './attack_imgs_mnist_origin/'
	dist, _ = get_knn_dist(path, I) 
	print("origin knn:", dist.item())
	
	path = './attack_imgs_mnist_mbdist/'
	dist, _ = get_knn_dist(path, I) 
	print("our knn:", dist.item())
