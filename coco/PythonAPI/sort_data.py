"""
This script is meant to gather a subset of animals from the 
MS_COCO data. Based on
https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoDemo.ipynb 

Authors: Judy and Nina
Date: April 2024
"""

#import numpy as np
from pycocotools.coco import COCO
#import matplotlib.pyplot as plt
import skimage.io as io



dataType = 'val2017'
annFile = '../annotations/instances_{}.json'.format(dataType)

coco = COCO(annFile)

cats = coco.loadCats(coco.getCatIds())
animal_categories = []
for cat in cats:
    if cat['supercategory'] == 'animal':
        animal_categories.append(cat['name'])


cats = coco.getCatIds(catNms=['cat'])
dogs = coco.getCatIds(catNms=['dog'])
imgIds_cats = coco.getImgIds(catIds = cats)
imgIds_dogs = coco.getImgIds(catIds = dogs)
print("num of cats: ", len(imgIds_cats))
print("num of dogs: ", len(imgIds_dogs))

cats_img_list = coco.loadImgs(imgIds_cats)
dogs_img_list = coco.loadImgs(imgIds_dogs)

i = 0
for img in cats_img_list:
    if (i % 1000) == 0:
        print("saving cat image:", i)
    file_name = img['file_name']
    I = io.imread(img['coco_url'])
    io.imsave(f'/cs/student/projects3/COMP0197/grp3/adl_groupwork/src/datasets/coco/val/cats/{file_name}', I)
    i = i + 1

i = 0
for img in dogs_img_list:
    if (i % 1000) == 0:
        print("saving dog image:", i)
    file_name = img['file_name']
    I = io.imread(img['coco_url'])
    io.imsave(f'/cs/student/projects3/COMP0197/grp3/adl_groupwork/src/datasets/coco/val/dogs/{file_name}', I)
    i = i + 1
