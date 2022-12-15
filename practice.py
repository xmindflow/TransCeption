import argparse
import logging
import os
import sys
import random
import numpy as np
from torchvision import transforms
import torch
import torch.backends.cudnn as cudnn
from networks.EfficientMISSFormer import EffMISSFormer
from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
from trainer import trainer_synapse
import warnings
warnings.filterwarnings('ignore')

from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from utils import test_single_volume
from torch.nn import functional as F
from datasets.dataset_synapse import Synapse_dataset, RandomGenerator

import matplotlib.pyplot as plt
import pandas as pd
import datetime


def trainer_synapse(args, model, snapshot_path):
    test_save_path = os.path.join(snapshot_path, 'test')
    os.makedirs(test_save_path, exist_ok=True)

    # Set logging
    logging.basicConfig(filename=snapshot_path+'/log.txt', level=logging.INFO,
    format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt="%H:%M:%S")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    # Set learning rate and other parameters
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu 

    x_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    y_transforms = transforms.ToTensor()

    # Load database
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",img_size=args.img_size,
                               norm_x_transform = x_transforms, norm_y_transform = y_transforms)
    print("The length of train set is: {}".format(len(db_train)))
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    db_test = Synapse_dataset(base_dir=args.test_path, split="test_vol", list_dir=args.list_dir, img_size=args.img_size)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)




parser = argparse.ArgumentParser()
parser.add_argument('--root_path',type=str,default='/images/PublicDataset/Transunet_synaps/project_TransUNet/data/Synapse/train_npz')
#blabla set all of the argument
args = parser.parse_args()

if __name__ == "__main__":
    # step1: set env
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # step2: check if the training is deterministic
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    
    #step3: load the data
    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': args.root_path,
            'list_dir': args.list_dir,
            'num_classes': 9,
        },
    }

    # batch_size and base_lr
    # output_dir
    net = EffMISSFormer(num_classes=args.num_classes).cuda(0)

    trainer = {'Synapse': trainer_synapse,}

    trainer[dataset_name](args, net, args.output_dir)# call trainer_synapse


