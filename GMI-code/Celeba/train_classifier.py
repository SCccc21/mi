import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as tvmodels
import matplotlib.pyplot as plt
import numpy as np
import logging
import time
import shutil

import random
from classify import *
from utils import *
# from tensorboardX import SummaryWriter

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

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


def validate(val_loader, model, criterion):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = None

    model.eval()
    end = time.time()

    with torch.no_grad():
        for i, (imgs, one_hot, label) in enumerate(val_loader):
            loss_meter = AverageMeter()
            data_time.update(time.time() - end)
            imgs =  imgs.cuda()
            label = label.cuda()

            out = model.module.predict(imgs)

            loss_val = criterion(out, label)

            loss_meter.update(loss_val, imgs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % 10 == 0:
                logger.info(
                    'Test: [{}/{}] '
                    'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                    'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    'val_loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}).'.format(
                        i + 1,
                        len(val_loader),
                        data_time=data_time,
                        batch_time=batch_time,
                        loss_meter=loss_meter))

    logger.info(
        'Val result: Val Loss {loss_meter.avg:.3f}.'.format(loss_meter=loss_meter))

    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    
    return loss_meter.avg


if __name__ == "__main__":

    global args, logger, writer
    logger = get_logger()
    # writer = SummaryWriter('./model_classifier')
    file = "./config/classify" + ".json"
    args = load_json(json_file=file)
    logger.info(args)
    logger.info("=> creating model ...")
    
   
    best_loss_all = 1e9

    train_path = args['dataset']['train_file_path']
    val_path = args['dataset']['test_file_path']
    model_name = args['dataset']['model_name']
    lr = args[model_name]['lr']
    batch_size = args[model_name]['batch_size']
    epochs = args[model_name]['epochs']

    save_model_dir = "/home/sichen/models/target_model/" + model_name
    os.makedirs(save_model_dir, exist_ok=True)

    if model_name.startswith("VGG16"):
        model = VGG16(1000)
        model = torch.nn.DataParallel(model).cuda()
    elif model_name.startswith('IR152'):
        model = IR152(1000)
        model = torch.nn.DataParallel(model).cuda()
    elif model_name == "FaceNet":
        model = FaceNet(1000)
        path = '/home/sichen/models/target_model/backbone_ir50_ms1m_epoch120.pth'
        model = torch.nn.DataParallel(model).cuda()
        ckp = torch.load(path)
        # import pdb; pdb.set_trace()
        load_module_state_dict(model, ckp, add="module.feature.")


    # model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss().cuda()

    train_set, train_loader = init_dataloader(args, train_path, batch_size, mode="classify")
    val_set, val_loader = init_dataloader(args, val_path, batch_size, mode="classify")
    
    print("---------------------Training [%s]------------------------------" % model_name)

    for e in range(epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_meter = None
        end = time.time()
        model.train()
        
        # training  
        for i, (imgs, one_hot, label) in enumerate(train_loader):
            
            data_time.update(time.time() - end)
            current_iter = (e - 1) * len(train_loader) + i + 1
            max_iter = epochs * len(train_loader)

            x = imgs.cuda()
            # one_hot = one_hot.long().cuda()
            label = label.cuda()
            img_size = x.size(2)
            bs = x.size(0)

            out = model.module.predict(x)
            
            loss = criterion(out, label)
            # import pdb; pdb.set_trace()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
           
            loss_meter = AverageMeter()
            loss_meter.update(loss.detach().cpu().numpy(), bs)
            batch_time.update(time.time() - end)
            end = time.time()
            # calculate remain time
            remain_iter = max_iter - current_iter
            remain_time = remain_iter * batch_time.avg
            t_m, t_s = divmod(remain_time, 60)
            t_h, t_m = divmod(t_m, 60)
            remain_time = '{:02d}:{:02d}:{:02d}'.format(
                int(t_h), int(t_m), int(t_s))

            if (i + 1) % 10 == 0:
                logger.info('Epoch: [{}/{}][{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Remain {remain_time}.'.format(
                                e,
                                epochs,
                                i + 1,
                                len(train_loader),
                                batch_time=batch_time,
                                data_time=data_time,
                                remain_time=remain_time))
                logger.info('Train Loss {loss_meter.val:.4f} '.format(loss_meter=loss_meter))

        # writer.add_scalar('loss_train', loss_train, e)
        is_best = False
        
        # evalutate
        loss_val = validate(val_loader, model, criterion)
        is_best = loss_val < best_loss_all
        best_loss_all = min(loss_val, best_loss_all)
        filename = os.path.join(save_model_dir, 'model_latest.pth')
        torch.save(
            {
                'epoch': e,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_loss_all': best_loss_all
            }, filename)
        model.cuda()

        if is_best:
            shutil.copyfile(filename,
                            os.path.join(save_model_dir, 'model_best.pth'))

        if e % 10 == 0:
            shutil.copyfile(
                filename,
                save_model_dir + '/train_epoch_' + str(e) + '.pth')
    
    


