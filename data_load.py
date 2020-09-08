import os
import numpy as np
import csv
import random
from PIL import Image
import imgaug as ia
from imgaug import augmenters as iaa

# Configuration

IMG_HEIGHT = 256
IMG_WIDTH = 256
channels=1
shuffle_input_data_rows=True

dataset_dir = 'TNSCUI2020_train/'



# read csv files

def get_train_dataset():
    datalist = []

    with open(dataset_dir + 'train.csv', newline='') as csvFile:
        reader = csv.reader(csvFile)

        for row in reader:
            datalist.append([row[0], row[1]])

        del datalist[0]

        if shuffle_input_data_rows:
            random.shuffle(datalist)

    X_train = np.zeros((len(datalist), IMG_WIDTH, IMG_WIDTH,1), dtype=np.uint8)
    Y_train = np.zeros((len(datalist),), dtype=np.uint8)
    i=0
    for row in datalist:
        image_path=dataset_dir+'image/'+row[0]
        image_label=row[1]
        img=Image.open(image_path)
        img = img.convert('L')
        img=img.resize((IMG_WIDTH,IMG_HEIGHT))
        array=np.asarray(img)
        X_train[i,:,:,0]=array
        Y_train[i]=image_label
        i+=1

    return X_train,Y_train

def get_train_val_dataset(train_factor=0.7):
    X_train,Y_train=get_train_dataset()
    if len(X_train)!=len(Y_train):
        print('ERROR: 图片标签数据集长度不同')
        return
    train_len=int(len(X_train)*train_factor)
    X_train_new=X_train[:train_len]
    Y_train_new=Y_train[:train_len]
    X_val=X_train[train_len:]
    Y_val=Y_train[train_len:]
    return X_train_new,Y_train_new,X_val,Y_val

# def data_augmentation(X_train,Y_train):
#     seq = iaa.Sequential([
#         iaa.Fliplr(0.5),  # 0.5的概率水平翻转
#         iaa.Crop(percent=(0, 0.1)),  # random crops
#         # sigma在0~0.5间随机高斯模糊，且每张图纸生效的概率是0.5
#         iaa.Sometimes(0.5,
#                       iaa.GaussianBlur(sigma=(0, 0.5))
#                       ),
#         # 增大或减小每张图像的对比度
#         iaa.ContrastNormalization((0.75, 1.5)),
#         # 高斯噪点
#         iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
#         # 给每个像素乘上0.8-1.2之间的数来使图片变暗或变亮
#         # 20%的图片在每个channel上乘以不同的因子
#         iaa.Multiply((0.8, 1.2), per_channel=0.2),
#         # 对每张图片进行仿射变换，包括缩放、平移、旋转、修剪等
#         iaa.Affine(
#             scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
#             translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
#             rotate=(-25, 25),
#             shear=(-8, 8)
#         )
#     ], random_order=True)  # 随机应用以上的图片增强方法
#     print(X_train.shape[0])
#     for image in X_train:
#         new_image=seq.augment_image(image)
#
#         X_train[X_train.shape[0], :, :, 0] = array
#         X_train[]


