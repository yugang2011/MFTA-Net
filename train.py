import os
import time
import argparse
import numpy as np
import torch
import random
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import warnings
from utils.util import setup_logger, print_args
from models import Trainer


def init_dist(backend='nccl', **kwargs):  # 分布式训练的初始化
    """initialization for distributed training"""
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def set_random_seed(seed):  # 设置随机数种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    warnings.filterwarnings("ignore")  # 忽略警告信息
    parser = argparse.ArgumentParser(description='referenceSR Training')
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--name', default='MTH-net-BraTS-FOLD5', type=str,help='训练更长的epoch版本')
    parser.add_argument('--phase', default='train', type=str)

    # device setting
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    # network setting
    parser.add_argument('--net_name', default='HAT', type=str, help='RefNet | Baseline')
    parser.add_argument('--sr_scale', default=4, type=int)  # 上采样倍数4
    parser.add_argument('--input_nc', default=1, type=int)  # 输入通道为1
    parser.add_argument('--output_nc', default=1, type=int)  # 输出通道为1
    parser.add_argument('--nf', default=64, type=int)
    parser.add_argument('--num_nbr', default=1, type=int)

    # dataloader setting
    parser.add_argument('--dataset', default='BRATS', type=str)

    parser.add_argument('--train_data_lr', default=r'./train-fold5/T2-LR-NPY', type=str)
    parser.add_argument('--train_data_hr', default=r'./train-fold5/T2-HR-NPY', type=str)
    parser.add_argument('--train_data_ref', default=r'./train-fold5/T1-HR-NPY', type=str)

    parser.add_argument('--test_data_lr', default=r'./test-fold5/T2-LR-NPY', type=str)
    parser.add_argument('--test_data_hr', default=r'./test-fold5/T2-HR-NPY', type=str)
    parser.add_argument('--test_data_ref', default=r'./test-fold5/T1-HR-NPY', type=str)
    parser.add_argument('--testset', default='TestSet', type=str, help='TestSet')
    parser.add_argument('--valset', default='ValSet', type=str, help='ValSet')
    parser.add_argument('--save_test_root', default='generated', type=str)
    parser.add_argument('--crop_size', default=128, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--multi_scale', action='store_true')
    parser.add_argument('--data_augmentation', action='store_true')

    # optim setting
    parser.add_argument('--lr', default=1e-4, type=float)  # 初始的学习率
    parser.add_argument('--weight_decay', default=0, type=float)
    # parser.add_argument('--lr_step_size', default=25, type=int)
    # parser.add_argument('--lr_gamma', default=0.1, type=int)
    parser.add_argument('--start_iter', default=0, type=int)  # 开始的周期数
    parser.add_argument('--max_iter', default=30, type=int)  # 结束的周期数

    parser.add_argument('--loss_l1_1x', action='store_true', default=True)  # 感知损失

    parser.add_argument('--lambda_l1_1x', default=1, type=float)

    # parser.add_argument('--resume', default=r'D:\Project\HAT-pytorch\HAT_SRx4_ImageNet-pretrain.pth', type=str)
    parser.add_argument('--resume_1', default=r'', type=str)
    parser.add_argument('--resume', default=r'', type=str)
    parser.add_argument('--resume_optim', default=r'', type=str)
    parser.add_argument('--resume_scheduler', default=r'', type=str)

    # log setting 日志文件
    parser.add_argument('--log_freq', default=10, type=int)
    parser.add_argument('--vis_freq', default=50000, type=int)  # 该参数决定了输出信息的批量
    parser.add_argument('--save_epoch_freq', default=1, type=int)  # 100
    parser.add_argument('--test_freq', default=1, type=int)  # 100
    parser.add_argument('--save_folder', default=r'/public/home/tech6/papertiger/project/MTH-Net/weights', type=str)
    parser.add_argument('--vis_step_freq', default=100, type=int)
    parser.add_argument('--use_tb_logger', action='store_true', default=True)
    parser.add_argument('--save_test_results', action='store_true', default=True)
    parser.add_argument('--use_loss', action='store_true', default=True)

    # tensorboard 输出文件
    parser.add_argument('--tensorboard_log', default=r'/public/home/tech6/papertiger/project/MTH-Net/tensorboard', type=str)
    # for evaluate
    parser.add_argument('--ref_level', default=1, type=int)

    # setup training environment
    args = parser.parse_args()
    set_random_seed(args.random_seed)

    # setup training device
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 0:
        torch.cuda.set_device(args.gpu_ids[0])

    #  distributed training settings  分布式训练
    if args.launcher == 'none':  # disabled distributed training 分布式训练不可用
        args.dist = False
        args.rank = -1
        print('Disabled distributed training.')
    else:
        args.dist = True
        init_dist()
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
    # 保存日志文件
    args.save_folder = os.path.join(args.save_folder, args.name)
    args.vis_save_dir = os.path.join(args.save_folder,  'vis')
    args.snapshot_save_dir = os.path.join(args.save_folder,  'snapshot')
    log_file_path = args.save_folder + '/' + time.strftime('%Y%m%d_%H%M%S') + '.log'
    # 检查路径，并创建文件路径
    if args.rank <= 0:
        if os.path.exists(args.vis_save_dir) == False:
            os.makedirs(args.vis_save_dir)
        if os.path.exists(args.snapshot_save_dir) == False:
            os.mkdir(args.snapshot_save_dir)
        setup_logger(log_file_path)

    print_args(args)

    cudnn.benchmark = True  # 自动寻找最适合当前配置的高效算法，来达到优化运行效率

    # train model
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
