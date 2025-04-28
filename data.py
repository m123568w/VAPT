import os
import torch
from torch.utils.data import DataLoader

class ImageDataset(torch.utils.data.Dataset):
    
    def __init__(self, img_dir):
        self.imgs = []
        self.labels = []
        label_dict = {'aircraft_carrier': 0, 
                      'supply': 1, 
                      'passenger': 2,
                      'destroyer': 3,
                      'cruiser': 4}
        for label in os.listdir(img_dir):
            dir_path = os.path.join(img_dir, label)
            for img in os.listdir(dir_path):
                img_path = os.path.join(dir_path, img)
                img_label = label
                self.imgs.append(img_path)
                self.labels.append(label_dict[img_label])
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        return img, label
    
    @property
    def num_classes(self):
        return len(set(self.labels))