import csv
import os
import random
import shutil


dataset_dir=''
output_dir=''

testing_factor=1

datalist=[]



# read csv files
with open(dataset_dir + 'train.csv', newline='') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        datalist.append([row[0], row[1]])

    del datalist[0]

training_size=int(len(datalist)*testing_factor)

training_set=datalist[:training_size]
validation_set=datalist[training_size:]


for row in training_set:
    id=row[0]
    label=row[1]

    file_path=dataset_dir+'image/'+str(id)
    mask_path=dataset_dir+'mask/'+str(id)
    if label=='0':
        shutil.copy(file_path,output_dir+'train/image/0/')
        shutil.copy(file_path, output_dir + 'train/mask/0')
    else:
        shutil.copy(file_path,output_dir+'train/image/1/')
        shutil.copy(file_path, output_dir + 'train/mask/1')

for row in validation_set:
    id=row[0]
    label=row[1]

    file_path=dataset_dir+'image/'+str(id)
    mask_path=dataset_dir+'mask/'+str(id)
    if label=='0':
        shutil.copy(file_path,output_dir+'val/image/0/')
        shutil.copy(file_path, output_dir + 'val/mask/0')
    else:
        shutil.copy(file_path,output_dir+'val/image/1/')
        shutil.copy(file_path, output_dir + 'val/mask/1')
