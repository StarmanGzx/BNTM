from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np



def get_img_lbl(data_dir, class_num):

    img_list = []
    lbl_list = []
    for label in range(class_num):
        class_dir = os.path.join(data_dir, str(label))
        if os.path.isdir(class_dir):
            for img in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img)
                if os.path.isfile(img_path):
                    img_list.append(img_path)
                    lbl_list.append(label)

    return img_list, lbl_list

# transform_rare = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
         
class Rareplanes(Dataset):

    def __init__(self, data_dir='/data/Rareplanes', transform=None, train=True):
        
        self.is_train = train
        if self.is_train:
            self.split = 'train'
        else:
            self.split = 'val'

        self.classes = 7
        
        self.data_root = data_dir
        self.img_dir = os.path.join(self.data_root, self.split)

        self.images , self.labels = get_img_lbl(self.img_dir, self.classes)
        
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