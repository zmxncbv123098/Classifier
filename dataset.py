import os
import torchvision.transforms as transforms
import random
import cv2


class CustomDataset(object):
    def __init__(self, transform, labels):
        self.transform = transform
        self.labels = labels
        self.imgs = self.get_imgs()

    def __getitem__(self, idx):
        # load images
        img_path = os.path.join("category_imgs", self.imgs[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (32, 32))
        category_name, image_name = os.path.split(self.imgs[idx])

        return self.transform(img), self.labels[category_name]

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def get_imgs():
        result = list()
        amounts = dict()
        for cat_name in os.listdir(os.path.join("category_imgs")):
            amounts[cat_name] = len(os.listdir(os.path.join("category_imgs", cat_name)))
            for img_name in os.listdir(os.path.join("category_imgs", cat_name)):
                result.append(os.path.join(cat_name, img_name))

        max_amount = max(amounts.values())
        for cat_name in amounts.keys():
            cat_imgs_list = os.listdir(os.path.join("category_imgs", cat_name))
            for i in range(amounts[cat_name], max_amount):
                result.append(os.path.join(cat_name, cat_imgs_list[random.randint(0, len(cat_imgs_list) - 1)]))
        return result


def get_transform(train):
    t = []
    t.append(transforms.ToTensor())
    if train:
        t.append(transforms.RandomHorizontalFlip(0.5))
    t.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return transforms.Compose(t)


labels = {'bird': 1, 'cat': 2, 'dog': 3, 'horse': 4, 'sheep': 5}
