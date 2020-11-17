import os
import cv2
from pycocotools.coco import COCO


folder = 'annotations/'
filename = 'instances_train2017.json'
annFile = os.path.join(folder, filename)
coco = COCO(annFile)

too_small = 0
all_amount = 0
target_amount = 0
sizes = {}
for i in os.listdir(os.path.join('', 'category_imgs')):
    print(i, end=": ")
    target_amount = 0
    for j in os.listdir(os.path.join('category_imgs', i)):

        catIds = coco.getCatIds(catNms=[i])
        img = coco.loadImgs([int(j.split(".")[0])])[0]
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)

        area = 0
        for block in anns:
            bbox = block["bbox"]
            area += bbox[2] * bbox[3]

        height, width = cv2.imread(os.path.join('category_imgs', i, j), 0).shape

        if int((area / (height*width)) * 100) > 10:
            target_amount += 1
            all_amount += 1
            size = str(height)+"x"+str(width)
            try:
                sizes[size] += 1
            except KeyError:
                sizes[size] = 1
        else:
            too_small += 1

    print(target_amount)
print("All images:", all_amount)
print("Small object:", too_small)
for i in sizes.keys():
    if sizes[i] > 20:
        print(i, ':', sizes[i])

