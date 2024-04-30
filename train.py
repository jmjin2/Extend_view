import os
import datetime
import logging
import torch
from torch.utils.data import DataLoader
import argparse
import math
import time
import numpy as np
import random
import os
from os import path as osp
from models.model import Model
from data.dataset import MIVRecurrentDataset
from utils.logger import AvgTimer, init_tb_logger, get_root_logger, get_env_info

def init_tb_loggers(args):
    # initialize wandb logger before tensorboard logger to allow proper sync
    tb_logger = None
    tb_logger = init_tb_logger(log_dir=osp.join('.', 'tb_logger', args.model))
    return tb_logger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_path', type=str, default='data/train', help='input train image folder')
    parser.add_argument('--model', type=str, default='MultiviewSkipSR')
    parser.add_argument('--basicvsr_path', type=str, default="pretrained/basicVSR/BasicVSR_REDS4.pth")
    parser.add_argument('--spynet_path', type=str, default="pretrained/flownet/spynet_sintel_final-3d2a1287.pth")
    parser.add_argument('--resume_state_path', type=str)
    parser.add_argument('--num_feat', type=int, default=64)
    parser.add_argument('--num_block', type=int, default=15)
    parser.add_argument('--pretrained_path', type=str)
    parser.add_argument('--iteration', type=int, default=50000)
    args = parser.parse_args()

    seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    # resume

    resume_state_path = args.resume_state_path
    if resume_state_path is None:
        resume_state = None
    else:
        device_id = torch.cuda.current_device()
        resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
 
    # mkdir
    experiments_root = osp.join('.', 'experiments')
    experiments_root = osp.join(experiments_root, args.model)
    if resume_state is None:
        if osp.exists(experiments_root):
            new_name = experiments_root + '_archived_' + time.strftime('%Y%m%d_%H%M%S', time.localtime())
            print(f'Path already exists. Rename it to {new_name}', flush=True)
            os.rename(experiments_root, new_name)
        os.makedirs(experiments_root, exist_ok=True)
        log_path = osp.join('.', 'tb_logger', args.model)
        if osp.exists(log_path):
            new_logname = log_path + '_archived_' + time.strftime('%Y%m%d_%H%M%S', time.localtime())
            print(f'Path already exists. Rename it to {new_logname}', flush=True)
            os.rename(log_path, new_logname)
        os.makedirs(log_path, exist_ok=True)

    # log
    log_file = osp.join(experiments_root, f"train_{args.model}_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}.log")
    logger = get_root_logger(log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    # initialize wandb and tb loggers
    tb_logger = init_tb_loggers(args)

    # create dataloader
    train_set = MIVRecurrentDataset()
    train_loader = DataLoader(train_set,
                              batch_size=1,
                              shuffle=True,
                              num_workers=6,
                              drop_last=True,
                              pin_memory=True)
    
    num_iter_per_epoch = len(train_set)
    total_iters = args.iteration
    total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
    logger.info('Training statistics:'
                        f'\n\tNumber of train images: {len(train_set)}'
                        f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                        f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')
    
    # create model
    model = Model(args)

    if resume_state:  # resume training
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        start_epoch = 0
        current_iter = 0

    start_iter = current_iter

    # training
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()

    for epoch in range(start_epoch, total_epochs + 1):
        for i, train_data in enumerate(train_loader):
            message = ""
            data_timer.record()
            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            model.update_learning_rate(current_iter, warmup_iter=-1)

            # training
            model.feed_data(train_data)
            model.optimize_parameters(current_iter)
            iter_timer.record()
            if current_iter == 1:
                epoch_start_time = time.time()

            # log
            if current_iter % 100 == 0:
                lrs = model.get_current_learning_rate()
                iter_time = iter_timer.get_avg_time()
                data_time= data_timer.get_avg_time()
                message = (f'[{args.model}..][epoch:{epoch:3d}, iter:{current_iter:8,d}, lr:(')
                for v in lrs:
                    message += f'{v:.3e},'
                    message += ')] '

                total_time = time.time() - epoch_start_time
                time_sec_avg = total_time / (current_iter - start_iter + 1)
                eta_sec = time_sec_avg * (total_iters - current_iter - 1)
                eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                message += f'[eta: {eta_str}, '
                message += f'time (data): {iter_time:.3f} ({data_time:.3f})] '

                logger.info(message)

            # save models and training states
            if current_iter % 5000 == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)
            
            data_timer.start()
            iter_timer.start()
            

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest

    if tb_logger:
        tb_logger.close()

if __name__ == '__main__':
    main()