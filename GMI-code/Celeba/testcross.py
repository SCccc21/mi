import os
import time
import utils
import torch
import dataloader
import torchvision
from utils import *
from torch.nn import BCELoss
from torch.autograd import grad
import torchvision.utils as tvls
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F
from discri import DGWGAN, Discriminator, MinibatchDiscriminator
from generator import Generator
from classify import *
from tensorboardX import SummaryWriter
from datetime import datetime
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

def freeze(net):
    for p in net.parameters():
        p.requires_grad_(False) 

def unfreeze(net):
    for p in net.parameters():
        p.requires_grad_(True)

def gradient_penalty(x, y):
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape).cuda()
    z = x + alpha * (y - x)
    z = z.cuda()
    z.requires_grad = True

    o = DG(z)
    g = grad(o, z, grad_outputs = torch.ones(o.size()).cuda(), create_graph = True)[0].view(z.size(0), -1)
    gp = ((g.norm(p = 2, dim = 1) - 1) ** 2).mean()

    return gp

def log_sum_exp(x, axis = 1):
    m = torch.max(x, dim = 1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(1)), dim = axis))


save_img_dir = "/home/sichen/models/result/ffhq"
os.makedirs(save_img_dir, exist_ok=True)

dataset_name = "celeba"



if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
    # os.environ["CUDA_VISIBLE_DEVICES"] = '4, 5, 6, 7'
    global args
    
    file = "./config/" + dataset_name + ".json"
    args = load_json(json_file=file)

    file_path = args['dataset']['train_file_path']
    model_name = args['dataset']['model_name']
    lr = args[model_name]['lr']
    batch_size = args[model_name]['batch_size']
    z_dim = args[model_name]['z_dim']
    epochs = args[model_name]['epochs']
    n_critic = args[model_name]['n_critic']

    utils.print_params(args["dataset"], args[model_name])

    dataset, dataloader = utils.init_dataloader(args, file_path, batch_size, mode="gan")

    # for i, imgs in enumerate(dataloader):
    #     print('process batch {}'.format(i))
    #     imgs = imgs.cuda()
    #     save_tensor_images(imgs.detach(), os.path.join(save_img_dir, "ffhq_{}.png".format(i)), nrow = 8)




    