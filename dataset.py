import os
from PIL import Image
import torchvision.transforms as transforms


class CustomDataset(object):
    def __init__(self, root):
        self.root = root
        self.transform = transforms.Compose(
            [transforms.Resize((32, 32)),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # load all image files
        self.imgs = list()
        for cat_name in os.listdir(os.path.join(root, "category_imgs")):
            for img_name in os.listdir(os.path.join(root, "category_imgs", cat_name)):
                self.imgs.append(os.path.join(cat_name, img_name))

    def __getitem__(self, idx):
        # load images
        img_path = os.path.join(self.root, "category_imgs", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        category_name, image_name = os.path.split(self.imgs[idx])
        labels = {'bird': 1, 'cat': 2, 'dog': 3, 'horse': 4, 'sheep': 5}

        return self.transform(img), labels[category_name]

    def __len__(self):
        return len(self.imgs)
