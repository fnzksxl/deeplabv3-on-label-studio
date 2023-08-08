import config
import custom_transfomrs as tr
import deeplab_resnet
from train import train
from dataset import make_dataset
from utils import *

import os
import argparse
from PIL import Image

from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision import transforms

import matplotlib.pyplot as plt

device

def mask_to_polygons(mask):
    # Convert mask to CV_8UC1 format
    mask_cv = (mask * 255).astype(np.uint8)

    contours, _ = cv2.findContours(mask_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        epsilon = 0.001 * cv2.arcLength(contour, True)
        approx_polygon = cv2.approxPolyDP(contour, epsilon, True)
        polygons.append(approx_polygon)

    return polygons

def test(args,device):
    test_pics = os.listdir(args.test_dir)
    model = deeplab_resnet.DeepLabv3_plus(nInputChannels=3, n_classes=args.n_classes, os=16).to(device)
    model.load_state_dict(torch.load('./model/best.pth',map_location=device))

    preprocess = transforms.Compose([
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
    ])
    model.eval()
    for pic in test_pics:
        input_image = Image.open(os.path.join(args.test_dir,pic))
        input_tensor = preprocess(input_image).unsqueeze(0)  # Add a batch dimension
        image_size = input_image.size
        
        with torch.no_grad():
            output = model(input_tensor)[0]
        
        object_masks=[]
        mask = output.argmax(0).cpu().numpy()
        
        to_Seperate_mask = mask.astype(np.uint8)
        for label in np.unique(mask):
            if label == 0:
                continue
            
            label_mask = (to_Seperate_mask == label).astype(np.uint8)

            contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(len(contours))
            for contour  in contours:                
                object_mask = np.zeros_like(mask,dtype=np.uint8)
                cv2.drawContours(object_mask,[contour],0,1,-1)
                object_masks.append(object_mask)

            for i, object_mask in enumerate(object_masks):
                plt.figure()
                plt.imshow(object_mask * 255, cmap='gray')
                plt.title(f'Object {i+1}')
                plt.show()

            polygons = mask_to_polygons(mask)
            image_np = np.array(input_image)

            plt.figure(figsize=(8, 8))
            plt.imshow(image_np)

            for polygon in polygons:
                plt.plot(polygon[:, 0, 0], polygon[:, 0, 1], 'r-', linewidth=2)

            plt.axis('off')
            plt.show()

            mask_image = Image.fromarray(mask.astype('uint8'))
            mask_image.save(os.path.join(args.output_dir,"mask.jpg"))


def data(args):
    image_path_lst=os.listdir(args.data_dir+'/image')
    mask_path_lst=os.listdir(args.data_dir+'/mask')
    image_path_lst.sort()
    mask_path_lst.sort()
    print(f"Whole Image Length : {len(image_path_lst)}")
    print(f"Whole Label Length : {len(mask_path_lst)}")

    train_val_rate = args.train_val_rate
    ceil = int(len(image_path_lst)*train_val_rate)
    
    print(f"Training Data Length : {ceil} ")
    print(f"Validation Data Length : {len(image_path_lst)-ceil}")

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

    print(f"Dataset Ready -")

    return traindata,valdata

def init_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--param', type=str, default='default value', help='help text')
    parser.add_argument('--mode', type=str, required=True, help='Traing or Demo?')
    parser.add_argument('--test_dir', type=str, default='./test_data', help='The folder you want to test')
    # Data configs
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--train_val_rate', type=float, default=0.8)
    parser.add_argument('--model_dir', type=str, default='./model')
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
    parser.add_argument('--data_dir', type=str, default='./', help='root directory of data')
    parser.add_argument('--output_dir', type=str, default='./')
    
    # Initialize args
    return parser.parse_args()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Current Device >>> " + device)
    print() 
    args = init_args()
    if args.mode == 'train':
        traindata,valdata = data(args)
        train(traindata,valdata,args)
    elif args.mode == 'test':
        test(args,device)
