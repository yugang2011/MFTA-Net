import os
import time
import logging
import itertools
import numpy as np
from PIL import Image
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision

from torch.utils.data import DataLoader
from collections import OrderedDict
import importlib

from utils import calculate_PSNR_SSIM
from models.modules import define_G
import SimpleITK as sitk



class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        self.augmentation = args.data_augmentation
        self.device = torch.device('cuda' if len(args.gpu_ids) != 0 else 'cpu')
        args.device = self.device

        # init dataloader
        testset_ = getattr(importlib.import_module('dataloader.dataset'), args.testset, None)
        self.test_dataset = testset_(self.args)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=1, num_workers=args.num_workers, shuffle=False)

        # init network
        self.net = define_G(args)
        if args.resume:
            self.load_networks('net', self.args.resume)

        if args.rank <= 0:
            logging.info('----- generator parameters: %f -----' % (sum(param.numel() for param in self.net.parameters()) / (10**6)))


    def prepare(self, batch_samples):
        for key in batch_samples.keys():
            if 'name' not in key and 'pad_nums' not in key:
                batch_samples[key] = Variable(batch_samples[key].to(self.device), requires_grad=False)
        return batch_samples

    def train(self):
        if self.args.rank <= 0:
            logging.info('training on  ...' + self.args.dataset)
            logging.info('%d training samples' % (self.train_dataset.__len__()))
            logging.info('the init lr: %f' % (self.args.lr))
        steps = 0
        self.net.train()

        if self.args.use_tb_logger:  # 写入tensorborad日志文件
            self.tensorboard_log = self.args.tensorboard_log
            if not os.path.exists(self.tensorboard_log):
                os.makedirs(self.tensorboard_log)
            if self.args.rank <= 0:
                tb_logger = SummaryWriter(log_dir=os.path.join(self.tensorboard_log, self.args.name))

        self.best_psnr = 0
        self.augmentation = False  # disenable data augmentation to warm up the encoder
        for i in range(self.args.start_iter, self.args.max_iter):
            self.scheduler.step()
            logging.info('current_lr: %f' % (self.optimizer_G.param_groups[0]['lr']))
            t0 = time.time()
            self.net.train()
            for j, batch_samples in enumerate(self.train_dataloader):
                log_info = 'epoch:%03d step:%04d  ' % (i, j)

                # prepare data
                batch_samples = self.prepare(batch_samples)
                LR = batch_samples['LR']

                # forward
                output = self.net(LR)
                # optimization
                loss = 0
                self.optimizer_G.zero_grad()

                if self.args.loss_l1_1x:
                    l1_loss1 = self.criterion(output[0], batch_samples['HR'])
                    l1_loss1 = l1_loss1
                    loss += l1_loss1
                    log_info += 'loss_SR_L1:%.06f ' % (l1_loss1.item())
                    l1_loss2 = self.criterion(output[1], batch_samples['Ref'])
                    l1_loss2 = l1_loss2
                    loss += l1_loss2
                    log_info += 'loss_SY_L1:%.06f ' % (l1_loss2.item())

                loss_l2_sr = self.criterion(output[2], batch_samples['HR3'])
                loss += loss_l2_sr
                log_info += 'loss_SR_L2:%.06f ' % (loss_l2_sr.item())
                loss_l3_sr = self.criterion(output[3],batch_samples['HR2'])
                loss += loss_l3_sr
                log_info += 'loss_SR_L3:%.06f ' % (loss_l3_sr.item())

                loss_l2_sy = self.criterion(output[5], batch_samples['Ref2'])
                loss += loss_l2_sy
                log_info += 'loss_SY_L2:%.06f ' % (loss_l2_sy.item())

                loss_l3_sy = self.criterion(output[4], batch_samples['Ref3'])
                loss += loss_l3_sy
                log_info += 'loss_SY_L3:%.06f ' % (loss_l3_sy.item())

                loss.backward()
                self.optimizer_G.step()

                # print information
                if j % self.args.log_freq == 0:
                    t1 = time.time()
                    log_info += 'aug:%s ' % str(self.augmentation)
                    log_info += '%4.6fs/batch' % ((t1-t0)/self.args.log_freq)
                    if self.args.rank <= 0:
                        logging.info(log_info)
                    t0 = time.time()

                # write tb_logger
                if self.args.use_tb_logger:
                    if steps % self.args.vis_step_freq == 0:
                        if self.args.rank <= 0:
                            if self.args.loss_l1_1x:
                                tb_logger.add_scalar('loss_sr', l1_loss1.item(), steps)
                                tb_logger.add_scalar('loss_sy', l1_loss2.item(), steps)
                steps += 1

            if True:
                    self.args.phase = 'eval'
                    psnr, ssim, pstd, sstd = self.test()
                    logging.info('SR psnr:%.06f ssim:%.06f  SY psnr:%.06f ssim:%.06f' % (psnr, pstd,ssim, sstd))
                    if psnr > self.best_psnr:
                        self.best_psnr = psnr
                        if self.args.rank <= 0:
                            logging.info('best_psnr:%.06f ' % (self.best_psnr))
                            logging.info('Saving state, epoch: %d iter:%d' % (i+1, 0))
                            self.save_networks('net', 'best')
                            self.save_networks('optimizer_G', 'best')
                            self.save_networks('scheduler', 'best')
                    self.args.phase = 'train'

        # end of training
        if self.args.rank <= 0:
            tb_logger.close()
            self.save_networks('net', 'final')
            logging.info('The training stage on %s is over!!!' % (self.args.dataset))

    def test(self):
        # save_path = os.path.join(self.args.save_folder, 'output_imgs')  # 测试输出保存路径

        self.net.eval()
        logging.info('start testing...')
        logging.info('%d testing samples' % (self.test_dataset.__len__()))
        num = 0
        PSNR = []
        SSIM = []
        PSNR_ = []
        SSIM_ = []

        with torch.no_grad():
            self.net.eval()
            for batch, batch_samples in enumerate(self.val_dataloader):
                batch_samples = self.prepare(batch_samples)
                HR = batch_samples['HR']
                Ref = batch_samples['Ref']
                LR = batch_samples['LR']

                output = self.net(LR)

                output_img = np.array(output[0].cpu().squeeze(0).squeeze(0))
                gt = np.array(HR.cpu().squeeze(0).squeeze(0))
                psnr = calculate_PSNR_SSIM.psnr(output_img, gt)
                ssim = calculate_PSNR_SSIM.ssim(output_img, gt)
                PSNR.append(psnr)
                SSIM.append(ssim)

                output_img_ = np.array(output[1].cpu().squeeze(0).squeeze(0))
                gt_ = np.array(Ref.cpu().squeeze(0).squeeze(0))
                psnr_ = calculate_PSNR_SSIM.psnr(output_img_, gt_)
                ssim_ = calculate_PSNR_SSIM.ssim(output_img_, gt_)
                logging.info('psnr_sr: %.6f    ssim_sr: %.6f  psnr_sy: %.6f    ssim_sy: %.6f' % (psnr, ssim,psnr_,ssim_))

                PSNR.append(psnr)
                SSIM.append(ssim)
                PSNR_.append(psnr_)
                SSIM_.append(ssim_)

                image_name = batch_samples['HR_name'][0]
                # path = os.path.join(save_path, image_name)
                # out_img = output[0].flip(dims=(0,)).clamp(0., 1.)
                # gt_img = HR[0].flip(dims=(0,))

                # torchvision.utils.save_image(torch.stack([out_img, gt_img]), path)
                logging.info('saving %d_th image: %s' % (num, image_name))
                num += 1

        psnr = np.mean(PSNR)
        ssim = np.mean(SSIM)
        psnr_ = np.mean(PSNR_)
        ssim_ = np.mean(SSIM_)

        PSNR_SR_STD = np.std(PSNR, ddof=1)
        SSIM_SR_STD = np.std(SSIM, ddof=1)

        PSNR_SY_STD = np.std(PSNR_, ddof=1)
        SSIM_SY_STD = np.std(SSIM_, ddof=1)


        logging.info('--------- average SR PSNR: %.06f,standard deviation:%.06f,SSIM:%.06f,standard deviation:%.06f'
                     ' SY PSNR: %.06f,standard deviation:%.06f, SSIM:%.06f, standard deviation:%.06f' % (
        psnr,PSNR_SR_STD, ssim,SSIM_SR_STD, psnr_,PSNR_SY_STD, ssim_, SSIM_SY_STD))

    def save_image(self, tensor, path):
        img = Image.fromarray(((tensor/2.0 + 0.5).data.cpu().numpy()*255).transpose((1, 2, 0)).astype(np.uint8))
        img.save(path)

    def load_networks(self, net_name, resume, strict=True):
        load_path = resume
        network = getattr(self, net_name)
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        load_net = torch.load(load_path, map_location=torch.device(self.device))

        model_dict = network.state_dict()
        pretrained_dict = {key: value for key, value in load_net['params_ema'].items() if
                           (key in model_dict and 'Prediction' not in key)}
        model_dict.update(pretrained_dict)

        network.load_state_dict(model_dict)

    def save_networks(self, net_name, epoch):
        network = getattr(self, net_name)
        save_filename = '{}_{}.pth'.format(net_name, epoch)
        save_path = os.path.join(self.args.snapshot_save_dir, save_filename)
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        state_dict = network.state_dict()
        if not 'optimizer' and not 'scheduler' in net_name:
            for key, param in state_dict.items():
                state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)
