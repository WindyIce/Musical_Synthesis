import os
from PIL import Image

test_dir='test/'
predict_dir='predict/'
output_dir='output/'

id2size=[]


test_paths=os.listdir(test_dir)

for file in test_paths:
    filename=os.path.join(test_dir,file)
    temp=filename.split('_')[1]
    id=int(temp.split('.')[0])
    image=Image.open(filename)
    img_size=image.size
    id2size.append([id,img_size])



predict_paths=os.listdir(predict_dir)

for row in id2size:
    id=row[0]
    image_size=row[1]
    filename=predict_dir+str(id)+'.PNG'
    print(filename)
    predict_img=Image.open(filename)
    output_img=predict_img.resize(image_size)
    output_img.save(output_dir+str(id)+'_output.PNG')






