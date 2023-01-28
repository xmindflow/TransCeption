import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset
from utils import test_single_volume
from networks.MSTr  import MSTransception
from trainer import trainer_synapse



parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/images/PublicDataset/Transunet_synaps/project_TransUNet/data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--volume_path', type=str,
                    default='/images/PublicDataset/Transunet_synaps/project_TransUNet/data/Synapse', help='root dir for data')
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
parser.add_argument('--is_savenii', type=bool, default=True, help="True, then the prediction can be saved")
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
parser.add_argument('--concat', type=str,  default="coord", help='normal--2d concat; 3d--3d concat')
parser.add_argument('--have_bridge', type=str,
                    default='original', help='None: no bridge; new:new bridge; original: original bridge')

parser.add_argument('--use_sa_config',type=int,  default=1, help='use_sa_config')
parser.add_argument('--sa_ker',type=int,  default=7, help='set kernel size for cbam')
parser.add_argument('--grad_clipping', type=bool, default=False, help='use grad clipping or not')
parser.add_argument('--use_scheduler', type=bool, default=True, help='True cos scheduler is used.')
parser.add_argument('--Stage_3or4',type=int,  default=3, help='setting the number of MS stages.')
parser.add_argument('--inter', type=str,  default="res", help='decide the interface in the msca-stage in MSViT_casa')
parser.add_argument('--br_config', type=int, default=2, help='choose the config for bridge attention for sequence.')
parser.add_argument('--weight_pth', type=str, default="./output/best_val.pth", help="set the path to the trained weight")
args = parser.parse_args()


if args.dataset == "Synapse":
    args.volume_path = os.path.join(args.volume_path, "test_vol_h5")
# config = get_config(args)


def inference(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol",img_size=args.img_size, list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return "Testing Finished!"


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'volume_path': args.volume_path,
            'list_dir': '/home/students/yiwei/yiwei_gitlab/MISSFormer-bridge/MISSFormer/lists/lists_Synapse',
            'num_classes': 9,
            'z_spacing': 1,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True
    if args.dil_conv:
        print('----------using dil conv: dil = 2--------')
    else:
        print('----------not using dil conv-------------')

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

    # net = Transception(num_classes=args.num_classes, head_count=1, dil_conv = args.dil_conv, token_mlp_mode="mix_skip").cuda(0)
    # net = MSTransception(num_classes=args.num_classes, head_count=args.head_count, dil_conv = args.dil_conv, token_mlp_mode="mix_skip", MSViT_config=args.MSViT_config, concat=args.concat, have_bridge=args.have_bridge, use_sa_config=args.use_sa_config,sa_ker=args.sa_ker,Stage_3or4=args.Stage_3or4, inter = args.inter).cuda()
    net = MSTransception(num_classes=args.num_classes, head_count=args.head_count, token_mlp_mode="mix_skip", MSViT_config=args.MSViT_config, concat=args.concat, have_bridge=args.have_bridge, Stage_3or4=args.Stage_3or4, br_ch_att_list=br_ch_att_list).cuda()

    #snapshot = os.path.join(args.output_dir, 'best_model.pth')
    # snapshot = f'/home/students/yiwei/yiwei_gitlab/EffFormer/outputs/transfilm_epoch_399.pth'
    #snapshot = f'/work/scratch/yiwei/last_week/MSTr_coord_silu_tfff/MSTr_coord_silu_tfff_8223_epoch_399.pth'
    # snapshot = f'/work/scratch/yiwei/last_week/accu_worse_in_sequence/MSTr_coord_ftft_accu/MSTr_coord_ftft_epoch_339_7705.pth'
    #snapshot = f'/work/scratch/yiwei/last_week/MSTr_coord_silu_wobr/MSTr_coord_silu_wobr_epoch_259_7643.pth'
    snapshot = args.weight_pth
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1)+"_wo_bridge")
    msg = net.load_state_dict(torch.load(snapshot))
    print(f"test model name: {args.model_name}")
    print(f"save prediction?: {args.is_savenii}")
    snapshot_name = snapshot.split('/')[-1]

    # log_folder = './test_log/test_log_'
    log_folder = os.path.join(args.output_dir, 'test_log') 
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = os.path.join(args.output_dir, "predictions")
        test_save_path = args.test_save_dir 
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None


    inference(args, net, test_save_path)


#  python test.py --dataset Synapse --base_lr 0.05 --model_name MSTr_8224_test --output_dir /work/scratch/yiwei/last_week/MSTr_github/ --weight_pth /work/scratch/yiwei/last_week/MSTr_coord_silu_tfff/MSTr_coord_silu_tfff_8223_epoch_399.pth