# Modified from https://github.com/hbzju/PiCO/blob/main/utils/cifar100.py

import os.path
import pickle
from typing import Any, Callable, Optional, Tuple
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as dsets

from .wide_resnet import WideResNet
from .utils_algo import generate_uniform_cv_candidate_labels, generate_hierarchical_cv_candidate_labels,\
                        generate_instancedependent_candidate_labels
from .cutout import Cutout
from .autoaugment import CIFAR10Policy, ImageNetPolicy
import random 


def generate_noise_labels(labels, partialY, noise_rate=0.0):

    partialY_new = [] # must define partialY_new
    for ii in range(len(labels)):
        label = labels[ii]
        plabel =  partialY[ii]
        noise_flag = (random.uniform(0, 1) <= noise_rate) # whether add noise to label
        if noise_flag:
            ## random choose one idx not in plabel
            houxuan_idx = []
            for ii in range(len(plabel)):
                if plabel[ii] == 0: houxuan_idx.append(ii)
            if len(houxuan_idx) == 0: # all category in partial label
                partialY_new.append(plabel)
                continue
            ## add noise in partial label
            newii = random.randint(0, len(houxuan_idx)-1)
            idx = houxuan_idx[newii]
            assert plabel[label] == 1, f'plabel[label] != 1'
            assert plabel[idx]   == 0, f'plabel[idx]   != 0'
            plabel[label] = 0
            # Dont add new label 
            # plabel[idx] = 1
            partialY_new.append(plabel)
        else:
            partialY_new.append(plabel)
    partialY_new = np.array(partialY_new)
    return partialY_new

def load_cifar100(args):
    
    test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], \
                                  std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    ])

    original_train = dsets.CIFAR100(root=args.data_dir, train=True, download=True)
    ori_data, ori_labels = original_train.data, torch.Tensor(original_train.targets).long()

    test_dataset = dsets.CIFAR100(root=args.data_dir, train=False, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size*4, shuffle=False, \
                                              num_workers=args.workers, \
                                              sampler=torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False))
    
    if args.hierarchical:
        partialY_matrix = generate_hierarchical_cv_candidate_labels('cifar100', ori_labels, args)
    else:
        if args.exp_type == 'rand':
            partialY_matrix = generate_uniform_cv_candidate_labels(args, ori_labels)
        elif args.exp_type == 'ins':
            # ori_data = torch.Tensor(original_train.data)
            # model = WideResNet(depth=28, num_classes=100, widen_factor=10, dropRate=0.3)
            # model.load_state_dict(torch.load('./pmodel/cifar100.pt'))
            # ori_data = ori_data.permute(0, 3, 1, 2)
            # partialY_matrix = generate_instancedependent_candidate_labels(model, ori_data, ori_labels)
            ori_data = torch.Tensor(original_train.data)
        #model = WideResNet(depth=28, num_classes=10, widen_factor=10, dropRate=0.3)
        #model.load_state_dict(torch.load('./pmodel/cifar10.pt'))
            ori_data = ori_data.permute(0, 3, 1, 2)
        #partialY_matrix = generate_instancedependent_candidate_labels(model, ori_data, ori_labels)
            with open('/raid/home/tusharpk/pll-ins-noise/Consistency_Regularization/ins_noise_cifar100_7.npy', 'rb') as f:
                partialY_matrix = np.load(f)
                noise_rate = 0.1 
                print("Receved noise rate ",noise_rate)
                partial_labels = generate_noise_labels(ori_labels,partialY_matrix ,noise_rate)
                print("loaded cifar100 instance wise partial labels")
                print('Average candidate num: ', np.mean(np.sum(partial_labels, axis=1)))
                # bingo_rate = np.sum(partial_labels[np.arange(len(partial_labels))], ori_labels == 1.0) / len(partial_labels)
                # print('Average noise rate: ', 1 - bingo_rate)
                print("added extra ooc noise in candidate set ")
                print("using valen's partial label")
        
            ori_data = original_train.data
            
    temp = torch.zeros(partialY_matrix.shape)
    temp[torch.arange(partialY_matrix.shape[0]), ori_labels] = 1
    
    if torch.sum(torch.tensor(partialY_matrix)* temp) == partialY_matrix.shape[0]:
        print('Partial labels correctly loaded !')
    else:
        print('Inconsistent permutation !')
    
    print('Average candidate num: ', partialY_matrix.sum(1).mean())
    
    partial_training_dataset = CIFAR100_Partialize(ori_data, partialY_matrix, ori_labels.float())
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(partial_training_dataset)
    
    partial_training_dataloader = torch.utils.data.DataLoader(
        dataset=partial_training_dataset, 
        batch_size=args.batch_size, 
        shuffle=(train_sampler is None), 
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )
    
    return partial_training_dataloader, torch.tensor(partialY_matrix), train_sampler, test_loader



class CIFAR100_Partialize(Dataset):
    def __init__(self, images, given_partial_label_matrix, true_labels):
        
        self.ori_images = images
        self.given_partial_label_matrix = given_partial_label_matrix
        self.true_labels = true_labels
        
        self.transform1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], \
                                  std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        ])

        self.transform2 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4, padding_mode='reflect'),
            CIFAR10Policy(),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),
            transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], \
                                  std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        ])
        

    def __len__(self):
        return len(self.true_labels)
        
    def __getitem__(self, index):
        
        each_image1 = self.transform1(self.ori_images[index])
        each_image2 = self.transform2(self.ori_images[index])
        each_label = self.given_partial_label_matrix[index]
        each_true_label = self.true_labels[index]
        
        return each_image1, each_image2, each_label, each_true_label, index

