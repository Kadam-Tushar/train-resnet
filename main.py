import os
import math
import time
import random
import shutil
import logging
import warnings
import argparse
import builtins
import numpy as np
import tensorboard_logger as tb_logger

import torch
import torch.nn 
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

from utils.resnet import *
from utils.utils_algo import *

from utils.fmnist import load_fmnist
from utils.svhn import load_svhn
from utils.cifar10 import load_cifar10
from utils.cifar100 import load_cifar100
from utils.miniImagenet import load_miniImagenet

from resnet_backbone import resnet_b

parser = argparse.ArgumentParser(description='PyTorch implementation training resnets for various datasets)')
parser.add_argument('--dataset', default='cifar10', type=str, 
                    choices=['fmnist', 'SVHN', 'cifar10', 'cifar100', 'miniImagenet'],
                    help='dataset name')

parser.add_argument('--exp-dir', default='./experiment', type=str,
                    help='experiment directory for saving checkpoints and logs')

parser.add_argument('--data-dir', default='./data', type=str)

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', choices=['resnet18','resnet34'],
                    help='network architecture (only resnet18 used)')

parser.add_argument('-j', '--workers', default=0, type=int,
                    help='number of data loading workers (default: 0)')

parser.add_argument('--epochs', default=50, type=int, 
                    help='number of total epochs to run')

parser.add_argument('--start-epoch', default=0, type=int, 
                    help='manual epoch number (useful on restarts)')

parser.add_argument('-b', '--batch-size', default=256, type=int,
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')

parser.add_argument('-lr_decay_epochs', type=str, default='99,199,299',
                    help='where to decay lr, can be a list')

parser.add_argument('-lr_decay_rate', type=float, default=0.1,
                    help='decay rate for learning rate')

parser.add_argument('--cosine', action='store_true', default=False,
                    help='use cosine lr schedule')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')

parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')

parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')

parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                    help='url used to set up distributed training')

parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--rate', default=0.4, type=float,help='-1 for feature, 0.x for random')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--cuda_VISIBLE_DEVICES', default='0', type=str, \
                        help='which gpu(s) can be used for distributed training')

parser.add_argument('--num-class', default=10, type=int,
                    help='number of class')


parser.add_argument('--low-dim', default=128, type=int,
                    help='embedding dimension')

parser.add_argument('--latent-dim', default=512, type=int,
                    help='latent embedding dimension')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_VISIBLE_DEVICES

torch.set_printoptions(precision=2, sci_mode=False)

logging.basicConfig(format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler('./result/result_'+ args.dataset + '_bs' + str(args.batch_size) + '_p' \
                            + '_seed' + str(args.seed) + '_dis_' + args.dist_url[-5:] + '.log'),
        logging.StreamHandler()
])
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_file_name='model_best.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_file_name)


def main():

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))

    if args.seed is not None:
        warnings.warn('You have chosen to seed training. This will turn on the CUDNN deterministic setting, which can slow down your training considerably! You may see unexpected behavior when restarting from checkpoints.')
        print()
    
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')
        print()
    
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])


    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    
    
    model_path = 'ds_{ds}_lr_{lr}_ep_{ep}_arch_{arch}_sd_{seed}_dis_{dis}'.format(
                                            ds=args.dataset,
                                            lr=args.lr,
                                            ep=args.epochs,
                                            arch=args.arch,
                                            seed=args.seed,
                                            dis=args.dist_url[-5:]
                                            )

    
    args.exp_dir = os.path.join(args.exp_dir, model_path)
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
    
    logging.info(args)
    print()
    
    ngpus_per_node = torch.cuda.device_count()
    
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        print("distibuted training is disabled! ")
        main_worker(args.gpu, ngpus_per_node, args)


        
def main_worker(gpu, ngpus_per_node, args):
    
    cudnn.benchmark = True
    model_path = 'ds_{ds}_lr_{lr}_ep_{ep}_arch_{arch}_sd_{seed}_dis_{dis}'.format(
                                            ds=args.dataset,
                                            lr=args.lr,
                                            ep=args.epochs,
                                            arch=args.arch,
                                            seed=args.seed,
                                            dis=args.dist_url[-5:]
                                            )
    args.gpu = gpu
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
        
    if args.gpu is not None:
        print("Use GPU: {} for training\n".format(args.gpu))
        
    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    
    if args.gpu==0:
        print("=> creating model '{}'\n".format(args.arch))
    if args.arch == 'resnet18':
        model = resnet_b.resnet(depth=20, n_outputs=args.num_class)
    
    if args.arch == 'resnet32':
        model = resnet_b.resnet(depth=32, n_outputs=args.num_class)


    # print(model)
    
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        #raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    print("optimizer done!")
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    if args.dataset == 'fmnist':
        train_loader, train_partialY_matrix, train_sampler, test_loader = load_fmnist(args)

    elif args.dataset == 'SVHN':
        train_loader, train_partialY_matrix, train_sampler, test_loader = load_svhn(args)
        
    elif args.dataset == 'cifar10':
        train_loader, train_partialY_matrix, train_sampler, test_loader = load_cifar10(args)

    elif args.dataset == 'cifar100':
        train_loader, train_partialY_matrix, train_sampler, test_loader = load_cifar100(args)
    
    elif args.dataset == 'miniImagenet':
        train_loader, train_partialY_matrix, train_sampler, test_loader = load_miniImagenet(args)
        print("miniiamgenet loaded!")
    else:
        raise NotImplementedError("You have chosen an unsupported dataset. Please check and try again.")

    if args.gpu==0:
        logging.info('\nAverage candidate num: {}\n'.format(train_partialY_matrix.sum(1).mean()))

    tempY = train_partialY_matrix.sum(dim=1).unsqueeze(1).repeat(1, train_partialY_matrix.shape[1])
    uniform_confidence = train_partialY_matrix.float() / tempY
    uniform_confidence = uniform_confidence.cuda()
    
    
    
    

    best_acc = 0
    loss_ce = nn.CrossEntropyLoss()
    for epoch in range(args.start_epoch, args.epochs):
    
        is_best = False
        
   
        
        

        acc_train_cls = train(train_loader, model, loss_ce, optimizer, epoch, args)

        acc_test, _ = test(model, test_loader, args, epoch)

        if acc_test.item() > best_acc:
            best_acc = acc_test.item()

        save_checkpoint({
                'epoch': epoch + 1,
                'model': model_path,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict()
            }, is_best=is_best, filename='{}/checkpoint.pth.tar'.format(args.exp_dir),
            best_file_name='{}/checkpoint_best.pth.tar'.format(args.exp_dir))
        
        if args.gpu==0:
            with open(os.path.join(args.exp_dir, 'result.log'), 'a+') as f:
                f.write('Epoch {}: Train_Acc {}, Test_Acc {}, Best_Acc {}. (lr:{})\n'.format(\
                 epoch, acc_train_cls.avg, acc_test.item(), best_acc, optimizer.param_groups[0]['lr']))

            logging.info('Epoch {}: Train_Acc {}, Test_Acc {}, Best_Acc {}. (lr:{})\n'.format(\
                  epoch, acc_train_cls.avg, acc_test.item(), best_acc, optimizer.param_groups[0]['lr']))


            
def train(train_loader, model,loss_ce,optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':1.2f')
    data_time = AverageMeter('Data', ':1.2f')
    acc_cls = AverageMeter('Acc@Cls', ':2.2f')
    loss_cls_log = AverageMeter('Loss@cls', ':2.2f')


    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, acc_cls, loss_cls_log],
        prefix="Epoch: [{}]".format(epoch)
    )

    model.train()
    
    end = time.time()
    
    for i, (images_1, images_2, labels, true_labels, index) in enumerate(train_loader):

        data_time.update(time.time() - end)
        
        X_1, X_2, Y, index = images_1.cuda(), images_2.cuda(), true_labels.cuda(), index.cuda()
        
        
        _,p1 = model(X_1)
        _,p2 = model(X_2)

        p = (p1+p2)/2.0
        
        
        loss = loss_ce(Y,p)
        
        loss_cls_log.update(loss.item())
       
        acc = accuracy(p, Y)
        acc_cls.update(acc[0].item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        if args.gpu==0:
            if (i % 50 == 0) or ((i + 1) % len(train_loader) == 0):
                logging.info('Epoch:[{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'A_cls {Acc_cls.val:.4f} ({Acc_cls.avg:.4f})\t'
                    'L_cls {Loss_cls.val:.4f} ({Loss_cls.avg:.4f})\t'.format(
                        epoch, i + 1, len(train_loader), batch_time=batch_time,
                        data_time=data_time, Acc_cls=acc_cls,
                        Loss_cls=loss_cls_log
                    )
                )
        
    return acc_cls, loss_cls_log



def test(model, test_loader, args, epoch):
    with torch.no_grad():

        print('\n=====> Evaluation...\n')       
        model.eval()    
        
        top1_acc = AverageMeter("Top1")
        top5_acc = AverageMeter("Top5")
        
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.cuda(), labels.cuda()
            
            _, outputs = model(images)
            
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            
            top1_acc.update(acc1[0])
            top5_acc.update(acc5[0])

            if args.gpu==0:
                if (batch_idx % 10 == 0) or ((batch_idx + 1) % len(test_loader) == 0):
                    logging.info(
                        'Test:[{0}/{1}]\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'\
                        .format(batch_idx + 1, len(test_loader), top1=top1_acc, top5=top5_acc)
                    )
        
        # average across all processes
        acc_tensors = torch.Tensor([top1_acc.avg, top5_acc.avg]).cuda(args.gpu)
        
        dist.all_reduce(acc_tensors)        
        
        acc_tensors /= args.world_size
        
        if args.gpu==0:
            logging.info('Top1 Accuracy is %.2f%%, Top5 Accuracy is %.2f%%\n'%(acc_tensors[0], acc_tensors[1]))          

    return acc_tensors[0], acc_tensors[1]
    


if __name__ == '__main__':
    main()


