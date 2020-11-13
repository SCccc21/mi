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

def save_center(I):
	feat_path = './feat/center.npy'
	img_folder = '/home/sichen/mi/GMI-code/Celeba/fid/private_domain_new/'
	center = torch.zeros(300, 512)
	idx = 900

	for i in range(300):
		aver_feat = torch.zeros(512)
		x = torch.zeros(12, 3, 64, 64)
		for j in range(12):
			name = img_folder + str(idx+j) + 'th_iden_' + str(i+1) + '.png'
			image = Image.open(name)
			image = TF.to_tensor(image)
			x[j, :, :, :] = torch.unsqueeze(image, 0)
		# import pdb; pdb.set_trace()
		print(i)
		feat, _ = I(low2high(x))
		center[i, :] = torch.mean(feat.detach().cpu(), dim=0)
		idx += 27
		torch.cuda.empty_cache()

	np.save("./feat/center.npy", center.numpy())
	print("Success!")

def save_knn(I):
	feat_path = './feat/feat.npy'
	info_path = "./feat/info.npy"
	img_folder = '/home/sichen/mi/GMI-code/Celeba/fid/private_domain_new/'

	knn = torch.zeros(300*12, 512)
	info = torch.zeros(300*12)
	idx = 900
	for i in range(300):
		x = torch.zeros(12, 3, 64, 64)
		
		for j in range(12):
			name = img_folder + str(idx+j) + 'th_iden_' + str(i+1) + '.png'
			image = Image.open(name)
			image = TF.to_tensor(image)
			x[j, :, :, :] = torch.unsqueeze(image, 0)
			info[12*i+j] = i
		print(i)
		feat, _ = I(low2high(x))
		knn[12*i:12*(i+1), :] = feat.detach().cpu()
		idx += 27

	np.save(feat_path, knn.numpy())
	np.save(info_path, info.numpy())
	print("Success!")


def get_center_dist(base_path, I):
	listOfFile = os.listdir(base_path)

	img_list = []
	label_list = []
	for entry in listOfFile:
		_, _, label, _ = os.path.splitext(entry)[0].strip().split('_')

		img = Image.open(base_path+entry)
		img = TF.to_tensor(img)
		img_list.append(img)
		label_list.append(int(label)-1)
		

		# if int(label) in [1,2,3,4,5,6,7,8,9,10]:
		# 	img = Image.open(base_path+entry)
		# 	img = TF.to_tensor(img)
		# 	img_list.append(img)
		# 	label_list.append(int(label)-1)

	# import pdb; pdb.set_trace()
	image = torch.stack(img_list, dim=0)
	print(image.shape)
	iden = torch.LongTensor(label_list)
	feat, _ = I(low2high(image))
	dist = calc_center(feat.detach(), iden, path='./feat')
	return dist, feat.detach()

def get_knn_dist(base_path, I):
	listOfFile = os.listdir(base_path)

	img_list = []
	label_list = []
	for entry in listOfFile:
		idx, _, _, label, seed = os.path.splitext(entry)[0].strip().split('_')
		if int(idx) == 0 and int(label) < 100: 
			img = Image.open(base_path+entry)
			img = TF.to_tensor(img)
			img_list.append(img)
			label_list.append(int(label)-1)
	
	image = torch.stack(img_list, dim=0)
	# print(image.shape)
	iden = torch.LongTensor(label_list)
	feat, _ = I(low2high(image))
	dist = calc_knn(feat.detach(), iden, path='./feat')

	return dist, feat.detach()

def psnr(img1, img2):
	mse = torch.mean((img1 - img2) ** 2)
	return 20 * torch.log10(255.0 / torch.sqrt(mse))

if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6,7'

	I = FaceNet(1000)
	ckp = torch.load('/home/sichen/models/target_model/backbone_ir50_ms1m_epoch120.pth')
	load_state_dict(I.feature, ckp)
	I = torch.nn.DataParallel(I).cuda()
	
	print("Loading Backbone Checkpoint ")
	# I.load_state_dict(ckp['state_dict'], strict=False)
	# save_center(I)
	save_knn(I)

	# path = './feat/origin/'
	# center_dist, feat_ori = get_knn_dist(path, I)
	# print("origin center dist:", center_dist)

	# path = './feat/mb/'
	# center_dist , feat_mb = get_knn_dist(path, I)
	# print("mb center dist:", center_dist)

	# # print(feat_ori - feat_mb)

	# path = './feat/mb_h/'
	# center_dist, _ = get_knn_dist(path, I)
	# print("mb+h center dist:", center_dist)

	# path = './feat/dist_mb/'
	# center_dist, _ = get_knn_dist(path, I)
	# print("mb+dist center dist:", center_dist)

	# path = './feat/dist_mb_h/'
	# center_dist, _ = get_knn_dist(path, I)
	# print("mb+dist+h center dist:", center_dist)

	path = './fid/fid_origin/'
	dist, _ = get_knn_dist(path, I) 
	print("origin knn:", dist)
	
	path = './fid/fid_dist_entropy/'
	dist, _ = get_knn_dist(path, I) 
	print("our knn:", dist)
