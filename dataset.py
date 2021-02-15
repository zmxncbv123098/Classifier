import os
import random

import torchvision.transforms as transforms
import cv2
from pycocotools.coco import COCO


class CustomDataset(object):
    def __init__(self, transform, labels, ann_filepath, redundancy):
        self.transform = transform
        self.labels = labels
        self.imgs = self.get_imgs(ann_filepath, redundancy)

    def __getitem__(self, idx):
        # load images
        img_path = os.path.join("category_imgs", self.imgs[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))  # 32, 32
        category_name, image_name = os.path.split(self.imgs[idx])

        return self.transform(img), self.labels[category_name]

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def get_imgs(filepath, redundancy):
        coco = COCO(filepath)

        result = list()
        amounts = dict()
        for cat_name in os.listdir(os.path.join("category_imgs")):
            amounts[cat_name] = len(os.listdir(os.path.join("category_imgs", cat_name)))
            for img_name in os.listdir(os.path.join("category_imgs", cat_name)):

                catIds = coco.getCatIds(catNms=[cat_name])
                img = coco.loadImgs([int(img_name.split(".")[0])])[0]
                annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
                anns = coco.loadAnns(annIds)

                area = 0
                for block in anns:
                    bbox = block["bbox"]
                    area += bbox[2] * bbox[3]

                height, width = cv2.imread(os.path.join('category_imgs', cat_name, img_name), 0).shape
                if int((area / (height * width)) * 100) > redundancy:
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
        t.append(transforms.RandomHorizontalFlip())
    t.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))  # 0.5
    return transforms.Compose(t)


labels = {'bird': 0, 'cat': 1, 'dog': 2, 'horse': 3, 'sheep': 4}
