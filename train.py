import data_load
import matplotlib.pyplot as plt
from tensorflow import keras

X_train,Y_train=data_load.get_train_dataset()

Y_train=keras.utils.to_categorical(Y_train,2)

batch_size = 64
epoch = 200

print('INFO: 数据集读取完成')
print(X_train.shape, Y_train.shape)

import unet,resnet

# model=unet.get_full_unet()

model=resnet.ResnetBuilder.build_resnet_18((1,256,256),2)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])





model.summary()

print('INFO: 模型构建完成')

history=model.fit(x=X_train,
                  y=Y_train,
                  batch_size=batch_size,
                  validation_split=0.3,
                  epochs=epoch)

print('INFO: 模型训练完成')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epoch)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
