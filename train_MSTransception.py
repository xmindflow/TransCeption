import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
# from networks.segformer import MySegFormer as ViT_seg
# from networks.EfficientMISSFormer import EffMISSFormer
# from networks.MSTransception import MSTransception
# from networks.MSTransceptionPlayCat import MSTransception
from networks.MSTr import MSTransception
from trainer import trainer_synapse
# from trainer_accu import trainer_synapse
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/images/PublicDataset/Transunet_synaps/project_TransUNet/data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--test_path', type=str,
                    default='/images/PublicDataset/Transunet_synaps/project_TransUNet/data/Synapse/test_vol_h5', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--output_dir', type=str, 
                    default='./output_v5',help='output dir')                   
parser.add_argument('--max_iterations', type=int,
                    default=90000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=400, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--num_workers', type=int,
                    default=4, help='num_workers')
parser.add_argument('--eval_interval', type=int,
                    default=20, help='eval_interval')
parser.add_argument('--model_name', type=str,
                    default='transfilm', help='model_name')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.05,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--z_spacing', type=int,
                    default=1, help='z_spacing')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
# parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')


parser.add_argument('--dil_conv', type=int,  default=1, help='Set if use dilation conv or not')
parser.add_argument('--inception_comb', type=str,  default="135", help='Set the combination of kernels in the inception module.')

parser.add_argument('--head_count', type=int,  default=8, help='number of head in attention module')
parser.add_argument('--MSViT_config', type=int,  default=2, help='Set which config to use')
parser.add_argument('--concat', type=str,  default="normal", help='normal--2d concat; 3d--3d concat')
parser.add_argument('--have_bridge', type=str,
                    default='None', help='None: no bridge; new:new bridge; original: original bridge para:para bridge')

parser.add_argument('--use_sa_config',type=int,  default=1, help='use_sa_config in cbam')
parser.add_argument('--sa_ker',type=int,  default=7, help='set kernel size for cbam')

parser.add_argument('--grad_clipping', type=bool, default=False, help='use grad clipping or not')
parser.add_argument('--use_scheduler', type=bool, default=False, help='True cos scheduler is used.')
parser.add_argument('--Stage_3or4',type=int,  default=3, help='setting the number of MS stages.')
parser.add_argument('--inter', type=str,  default="res", help='decide the interface in the msca-stage in MSViT_casa')
parser.add_argument('--num_sp',type=int,  default=0, help='setting the number of spatial aware attention in the bridge.')
parser.add_argument('--br_config', type=int, default=0, help='choose the config for bridge attention for sequence.')
args = parser.parse_args()

# config = get_config(args)


if __name__ == "__main__":  
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    torch.cuda.empty_cache()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': args.root_path,
            'list_dir': args.list_dir,
            'num_classes': 9,
        },
    }

    if args.batch_size != 24 and args.batch_size % 5 == 0:
        args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    if args.dil_conv:
        print('----------using dil conv: dil = 2--------')
    else:
        print('----------not using dil conv-------------')

    # print('\n inception combination' + args.inception_comb)
    # args.use_scheduler = False
   
    print(f'using bridge: {args.have_bridge}')

    print(f'use_scheduler:{args.use_scheduler}')
    print(f"use concat module {args.concat}")
  
    if args.br_config == 0:
        print('In bridge, 4 spatial attention')
        br_ch_att_list = [False, False, False, False]
    elif args.br_config == 1:
        print('In bridge, 4 channel attention')
        br_ch_att_list = [True, True, True, True]
    elif args.br_config == 2:
        print('In bridge, c s s s')
        br_ch_att_list = [True, False, False, False]
    elif args.br_config == 3:
        print('In bridge, s c s c')
        br_ch_att_list = [False, True, False, True]
    else:
        print('In bridge, c s c s')
        br_ch_att_list = [True, False, True, False]

    # net = Transception(num_classes=args.num_classes, head_count=1, dil_conv = args.dil_conv, token_mlp_mode="mix_skip", inception=args.inception_comb).cuda(0)
    # MSTransception.py
#    net = MSTransception(num_classes=args.num_classes, head_count=args.head_count, dil_conv = args.dil_conv, token_mlp_mode="mix_skip", MSViT_config=args.MSViT_config, concat=args.concat, have_bridge=args.have_bridge).cuda()
    # MSTransception_playCat.py
 
    net = MSTransception(num_classes=args.num_classes, head_count=args.head_count, token_mlp_mode="mix_skip", MSViT_config=args.MSViT_config, concat=args.concat, have_bridge=args.have_bridge, Stage_3or4=args.Stage_3or4, br_ch_att_list=br_ch_att_list).cuda()


    trainer = {'Synapse': trainer_synapse,}
    trainer[dataset_name](args, net, args.output_dir)
