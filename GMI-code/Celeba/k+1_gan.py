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
from discri import DGWGAN, Discriminator
from generator import Generator
from classify import *

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

save_img_dir = "/home/sichen/models/result/imgs_improved_celeba_gan"
save_model_dir= "/home/sichen/models/improvedGAN"
os.makedirs(save_model_dir, exist_ok=True)
os.makedirs(save_img_dir, exist_ok=True)

dataset_name = "celeba"

log_path = "./attack_logs"
os.makedirs(log_path, exist_ok=True)
log_file = "GAN.txt"
utils.Tee(os.path.join(log_path, log_file), 'w')



if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '4, 5, 6, 7'
    
    file = "./config/" + dataset_name + ".json"
    args = load_params(json_file=file)

    file_path = args['dataset']['train_file_path']
    model_name = args['dataset']['model_name']
    lr = args[model_name]['lr']
    batch_size = args[model_name]['batch_size']
    z_dim = args[model_name]['z_dim']
    epochs = args[model_name]['epochs']
    n_critic = args[model_name]['n_critic']
    unlabel_weight = args[model_name]['unlabel_weight']

    model_name_T = "VGG16"
    path_T = '/home/sichen/models/target_model/' + model_name_T + '/model_best.pth'

    if model_name_T.startswith("VGG16"):
        T = VGG16(1000)
    elif model_name_T.startswith('IR152'):
        T = IR152(1000)
    elif model_name_T == "FaceNet64":
        T = FaceNet64(1000)

    T = torch.nn.DataParallel(T).cuda()
    ckp_T = torch.load(path_T)
    T.load_state_dict(ckp_T['state_dict'], strict=False)

    print("---------------------Training [%s]------------------------------" % model_name)
    utils.print_params(args["dataset"], args[model_name])

    dataset, dataloader = init_dataloader(args, file_path, batch_size, mode="gan")

    G = Generator(z_dim)
    DG = Discriminator(3, 64, 1000)
    
    G = torch.nn.DataParallel(G).cuda()
    DG = torch.nn.DataParallel(DG).cuda()

    dg_optimizer = torch.optim.Adam(DG.parameters(), lr=lr, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.CrossEntropyLoss().cuda()
    

    step = 0

    for epoch in range(epochs):
        start = time.time()
        _, unlabel_loader1 = init_dataloader(args, file_path, batch_size, mode="gan", iterator=True)
        _, unlabel_loader2 = init_dataloader(args, file_path, batch_size, mode="gan", iterator=True)

        for i, imgs in enumerate(dataloader):

            step += 1
            imgs = imgs.cuda()
            bs = imgs.size(0)
            # if bs < 64:
            #     continue
            x_unlabel = unlabel_loader1.next()
            unlabel2 = unlabel_loader2.next()
            
            freeze(G)
            unfreeze(DG)

            z = torch.randn(bs, z_dim).cuda()
            f_imgs = G(z)

            y_prob = T(imgs)[-1]
            y = torch.argmax(y_prob, dim=1).view(-1)
            

            _, output_label = DG(imgs)
            _, output_unlabel = DG(x_unlabel)
            _, output_fake =  DG(f_imgs)
            # import pdb; pdb.set_trace()

            loss_lab = criterion(output_label, y)
            # loss_lab = torch.mean(torch.mean(log_sum_exp(output_label)))-torch.mean(torch.gather(output_label, 1, y.unsqueeze(1)))
            loss_unlab = 0.5*(torch.mean(F.softplus(log_sum_exp(output_unlabel)))-torch.mean(log_sum_exp(output_unlabel))+torch.mean(F.softplus(log_sum_exp(output_fake))))
            dg_loss = loss_lab + loss_unlab
            
            acc = torch.mean((output_label.max(1)[1] == y).float())
            
            # wd = output_label.mean() - output_fake.mean()  # Wasserstein-1 Distance
            # gp = gradient_penalty(imgs.data, f_imgs.data)
            # loss_regular = - wd + gp * 10.0
            
            dg_optimizer.zero_grad()
            dg_loss.backward()
            dg_optimizer.step()

            # train G

            if step % n_critic == 0:
                freeze(DG)
                unfreeze(G)
                z = torch.randn(bs, z_dim).cuda()
                f_imgs = G(z)
                mom_gen, output_fake = DG(f_imgs)
                mom_unlabel, _ = DG(unlabel2)

                mom_gen = torch.mean(mom_gen, dim = 0)
                mom_unlabel = torch.mean(mom_unlabel, dim = 0)
                
                # g_loss = torch.mean((mom_gen - mom_unlabel).abs())  # feature matching loss
                g_loss = - torch.mean(F.softplus(log_sum_exp(output_fake)))

                # logit_dg = DG(f_imgs)
                # g_loss = - logit_dg.mean()
                # import pdb; pdb.set_trace()
                
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

        end = time.time()
        interval = end - start
        
        print("Epoch:%d \tTime:%.2f\tG_loss:%.2f\t train_acc:%.2f" % (epoch, interval, g_loss, acc))
        if (epoch+1) % 10 == 0:
            z = torch.randn(32, z_dim).cuda()
            fake_image = G(z)
            save_tensor_images(fake_image.detach(), os.path.join(save_img_dir, "improved_result_image_{}.png".format(epoch)), nrow = 8)
        
        # torch.save({'state_dict':G.state_dict()}, os.path.join(save_model_dir, "improved_celeba_G.tar"))
        # torch.save({'state_dict':DG.state_dict()}, os.path.join(save_model_dir, "improved_celeba_D.tar"))

