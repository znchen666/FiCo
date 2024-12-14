from PIL import Image
import os
import glob
from PIL import Image
from imagecorruptions import corrupt
import numpy as np
item_list = ['000', '001', '002', '003', '004', '005', '006', '007', '008', '009',]
for type_cup in ['brightness','contrast','defocus_blur','gaussian_noise']:
# for type_cup in ['defocus_blur']:
    for _class_ in item_list:
        path_orginal = '/data/cifar10/test/' + _class_
        path = '/data/cifar10_5_'+ type_cup +'/test/' + _class_
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
            print("The new directory is created!")
        image_names = glob.glob(path_orginal + '/*.jpg')
        for image_name in image_names:
            path_to_image = image_name
            print(path_to_image)
            image = Image.open(path_to_image)
            image = np.array(image)
            corrupted = corrupt(image, corruption_name=type_cup, severity=5)
            im = Image.fromarray(corrupted)
            dir = '/data/cifar10_5_'+ type_cup +'/test/' \
                  + image_name.split('/')[-2] + '/' + image_name.split('/')[-1]
            print(dir)
            im.save(dir)


