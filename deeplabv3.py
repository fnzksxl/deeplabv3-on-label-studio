import config
import custom_transfomrs as tr
from train import train
from dataset import make_dataset
from utils import *

import os
import argparse

from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision import transforms

def data(args):
    image_path_lst=os.listdir(args.data_dir+'/image')
    mask_path_lst=os.listdir(args.data_dir+'/mask')
    image_path_lst.sort()
    mask_path_lst.sort()
    print(f"Whole Image Length : {len(image_path_lst)}")
    print(f"Whole Label Length : {len(mask_path_lst)}")

    train_val_rate = args.train_val_rate
    ceil = int(len(image_path_lst)*train_val_rate)
        
    train_image = image_path_lst[:ceil]
    val_image = image_path_lst[ceil:]
    train_label = mask_path_lst[:ceil]
    val_label = mask_path_lst[ceil:]
    

    composed_transforms_tr = transforms.Compose([
            tr.FixedResize(size=(512, 512)),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

    composed_transforms_ts = transforms.Compose([
        tr.FixedResize(size=(512, 512)),
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])
    
    traindata = make_dataset(train_image,train_label,'train',args,composed_transforms_tr)
    valdata = make_dataset(val_image,val_label,'val',args,composed_transforms_ts)

    return traindata,valdata

def init_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--param', type=str, default='default value', help='help text')

    # Data configs
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--train_val_rate', type=float, default=0.8)
    
    # Model configs
    parser.add_argument('--lr', type=float, default=1e-7)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--n_classes', type=int, default=2)

    # Trainer configs
    parser.add_argument('--epoch_size', type=int, default=10)
    parser.add_argument('--nTestInterval', type=int, default=10)
    parser.add_argument('--nAveGrad', type=int, default=1)
    parser.add_argument('--nEpochs', type=int, default=100)
    parser.add_argument('--snapshot', type = int, default=10)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)

    # Initialize args
    return parser.parse_args()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print() 
    args = init_args()
    traindata,valdata = data(args)
    train(traindata,valdata,args)