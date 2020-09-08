import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

train_dir=''
testing_dir=''

train_0_dir = os.path.join(train_dir, '0')  # directory with our training cat pictures
train_1_dir = os.path.join(train_dir, '1')  # directory with our taraining dog pictures

num_0_tr = len(os.listdir(train_0_dir))
num_1_tr = len(os.listdir(train_1_dir))

num_test = len(os.listdir(testing_dir))

total_train = num_0_tr + num_1_tr
total_test = num_test

print('total training 0 images:', num_0_tr)
print('total training 1 images:', num_1_tr)

print("--")
print("Total training images:", total_train)
print("Total validation images:", total_test)

print('INFO: 文件信息')

batch_size = 128
epochs = 15
IMG_HEIGHT = 256
IMG_WIDTH = 256
channels=1

train_image_generator = ImageDataGenerator(rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.5) # Generator for our training data
testing_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           color_mode='grayscale',
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='categorical')



test_data_gen = testing_image_generator.flow_from_directory(batch_size=batch_size,
                                                              color_mode='grayscale',
                                                              directory=testing_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode=None)




# print('INFO: 训练集数据格式'+str(train_data_gen.labels))

print('INFO: 数据集读取完毕')

# sample_training_images, _ = next(train_data_gen)
#
# # This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
# def plotImages(images_arr):
#     fig, axes = plt.subplots(1, 5, figsi  ze=(20,20))
#     axes = axes.flatten()
#     for img, ax in zip( images_arr, axes):
#         ax.imshow(img)
#         ax.axis('off')
#     plt.tight_layout()
#     plt.show()
#
# plotImages(sample_training_images[:5])

from model_unet import unet,unet_2
from model_ResNet import get_ResNet_classifier
import unet
import resnet

model = resnet.ResnetBuilder.build_resnet_18((channels, IMG_WIDTH, IMG_HEIGHT), 2)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# model=unet.get_full_unet()

model.summary()

print('INFO: 模型构建完毕')

history = model.fit(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
)



print('INFO: 模型训练完成')

import datetime

i=datetime.datetime.now()
save_str='resnet'
model.save('model/model_'+save_str+'_'+i.isoformat()+'.h5')

print('模型保存完成')

acc = history.history['accuracy']

loss=history.history['loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

result=model.predict(test_data_gen)
model.predict
print(result)
