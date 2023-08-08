from PIL import Image
import argparse

from torch.utils.data import Dataset,DataLoader
    
class DeepLabDataset(Dataset):
    def __init__(self,images,labels,transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        
        return len(self.images)
    
    def __getitem__(self,idx):
        _img,_label = self._make_img_gt_point_pair(idx) 
        sample = {'image' : _img, 'label' : _label}
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample
    
    def _make_img_gt_point_pair(self,idx):
        _img = Image.open(self.images[idx]).convert('RGB')
        _label =Image.open(self.labels[idx])
        
        return _img,_label
    

def make_dataset(images,labels,mode,args,transform=None):
    dataset = DeepLabDataset(images,labels,transform)
    if mode == 'train':
      return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    elif mode == 'val':
        return DataLoader(dataset, batch_size=args.batch_size,shuffle=False, num_workers=0)
