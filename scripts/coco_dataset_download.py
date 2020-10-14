from pycocotools.coco import COCO
import os
import requests


folder = 'annotations/'
filename = 'instances_train2017.json'
annFile = os.path.join(folder, filename)

coco = COCO(annFile)
cats = coco.loadCats(coco.getCatIds())
nms = []
for cat in cats:
    if cat['supercategory'] == 'animal':
        nms.append(cat['name'])

print(" ".join(nms))
for name in nms[0:5]:
    print(name)
    current = 0
    catIds = coco.getCatIds(catNms=[name])
    imgIds = coco.getImgIds(catIds=catIds)
    img = coco.loadImgs(imgIds)
    print(len(img))
    path = "./" + name
    try:
        os.mkdir(path)
    except OSError as error:
        print(error)

    listLS = os.listdir(path)
    for i in img:
        if i['file_name'] not in listLS:
            p = requests.get(i['coco_url'])
            img_path = path + "/" + i['file_name']
            current += 1
            out = open(img_path, "wb")
            out.write(p.content)
            out.close()
            if current % 100 == 0:
                print(current)
