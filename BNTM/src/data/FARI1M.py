from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np



def get_img_lbl(data_txt_path):
    class_to_idx = {
    'A220': 0,
    'A321': 1,
    'A330': 2,
    'A350': 3,
    'ARJ21': 4,
    'Boeing737': 5,
    'Boeing747': 6,
    'Boeing777': 7,
    'Boeing787': 8,
    'C919': 9
    }         
 
    img_list = []
    lbl_list = []
    with open(data_txt_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip().split()

        img_path = os.path.join('/data/FAIR1M1_aircraft/train', line[1], line[0])
        
        img_list.append(img_path)
        lbl_list.append(class_to_idx[line[1]])

    return img_list, lbl_list



class FAIRI1M(Dataset):

    def __init__(self, data_dir='/data/FAIR1M1_aircraft', transform=None, is_train=True):
        
        self.data_root = data_dir
        self.classes = 10

        self.is_train = is_train
        if self.is_train:
            self.data_txt_path = os.path.join(self.data_root, 'train.txt')
        else:
            self.data_txt_path = os.path.join(self.data_root, 'val.txt')


        self.images , self.labels = get_img_lbl(self.data_txt_path)
        
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    
    def show_batch(self, img_batch, class_batch):

        for i in range(img_batch.shape[0]):
            ax = plt.subplot(1, img_batch.shape[0], i + 1)
            title_str = self.map_class(int(class_batch[i]))
            img = np.transpose(img_batch[i, ...], (1, 2, 0))
            ax.imshow(img)
            ax.set_title(title_str.__str__(), {'fontsize': 5})
            plt.tight_layout()


# transform_rare = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])