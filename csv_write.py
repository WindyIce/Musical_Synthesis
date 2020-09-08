
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import *

import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import csv




model_path=''

test_path=''

output_path=''

model=load_model(model_path)
data_rows=[['ID','CATE']]

for file in os.listdir(test_path):
    image=Image.open(os.path.join(test_path,file))
    image = image.convert('L')
    image=image.resize((256,256))
    to_pre=np.zeros((1, 256, 256,1))
    array = np.asarray(image)
    array=array/255.0
    to_pre[0,:,:,0]=array
    label=model.predict([to_pre])[0]
    label=np.around(label)

    string=file.split('_')[1]
    data_rows.append([string,label[1]])

with open(output_path,'w') as file:
    f_csv = csv.writer(file)
    f_csv.writerow(data_rows)





