import os
from PIL import Image
import torchvision.transforms as transforms
import random


class CustomDataset(object):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        # load all image files
        self.imgs = list()
        amounts = dict()
        for cat_name in os.listdir(os.path.join(root, "category_imgs")):
            amounts[cat_name] = len(os.listdir(os.path.join(root, "category_imgs", cat_name)))
            for img_name in os.listdir(os.path.join(root, "category_imgs", cat_name)):
                self.imgs.append(os.path.join(cat_name, img_name))

        max_amount = max(amounts.values())
        for cat_name in amounts.keys():
            cat_imgs_list = os.listdir(os.path.join(root, "category_imgs", cat_name))
            for i in range(amounts[cat_name], max_amount):
                self.imgs.append(os.path.join(cat_name, cat_imgs_list[random.randint(0, len(cat_imgs_list) - 1)]))

    def __getitem__(self, idx):
        # load images
        img_path = os.path.join(self.root, "category_imgs", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        category_name, image_name = os.path.split(self.imgs[idx])
        labels = {'bird': 1, 'cat': 2, 'dog': 3, 'horse': 4, 'sheep': 5}

        return self.transform(img), labels[category_name]

    def __len__(self):
        return len(self.imgs)


def get_transform(train):
    t = []
    t.append(transforms.Resize((32, 32)))  # 224, 224
    t.append(transforms.ToTensor())
    if train:
        t.append(transforms.RandomHorizontalFlip(0.5))
    t.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return transforms.Compose(t)
