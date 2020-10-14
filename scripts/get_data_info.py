import os
import cv2

all_amount = 0
target_amount = 0
sizes = {}
for i in os.listdir(os.path.join('', '../category_imgs')):
    print(i, end=": ")
    target_amount = 0
    for j in os.listdir(os.path.join('../category_imgs', i)):
        target_amount += 1
        all_amount += 1

        height, width = cv2.imread(os.path.join('../category_imgs', i, j), 0).shape
        size = str(height)+"x"+str(width)
        try:
            sizes[size] += 1
        except KeyError:
            sizes[size] = 1

    print(target_amount)
print("All images:", all_amount)
for i in sizes.keys():
    if sizes[i] > 20:
        print(i, ':', sizes[i])

