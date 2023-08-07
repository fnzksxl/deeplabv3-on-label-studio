n_classes = 2
batch_size = 8 
nEpochs = 150
snapshot = 10 # {snapshot} 에포크마다 모델 저장
lr = 1e-7 # Learning Rate
wd = 5e-4 # Weight Decay
momentum = 0.9 # Momentum
epoch_size = 10 # {epoch_size} 에포크마다 lr 변경
nAveGrad = 1
train_val_rate = 0.8 # train 8 : 2 val
nTestInterval = 5
save_dir = 'saved_model'

import custom_transfomrs as tr
from torchvision import transforms

composed_transforms_tr = transforms.Compose([
        tr.FixedResize(size=(512, 512)),
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])

composed_transforms_ts = transforms.Compose([
    tr.FixedResize(size=(512, 512)),
    tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    tr.ToTensor()])