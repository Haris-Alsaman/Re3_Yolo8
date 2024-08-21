import os
import json
import numpy as np

# befor run this code you should have json file contain the label of the image
# if you have data for Yolo you can not use this code !
# images names in this code  is 000001.jpg to 001000.jpg 
# change code to your images names
# you can change code for your data format


# path to the directory containing images
image_dir = '/provided_small/20190926_183400_1_7'
# i take label from json file 
json_file = 'IR_label.json'  


# 
with open(json_file, 'r') as f:
    data = json.load(f)

exist_list = data['exist']
gt_rects = data['gt_rect']

#list to save the image names and labels
image_names = []
labels = []

# loop over the images. data format is 000001.jpg to 001000.jpg
for idx, (exist, gt_rect) in enumerate(zip(exist_list, gt_rects)):
    if exist:
        # إنشاء اسم الصورة
        image_name = f'{idx+1:06d}.jpg'  # صيغة الأسماء 000001.jpg إلى 001000.jpg
        image_path = os.path.join(image_dir, image_name)
        
        # افترض أن video_id هو 0 وأن track_id هو 0، ويمكنك تغييره بناءً على السيناريو الخاص بك
        video_id = 0
        track_id = 0
        
        # إضافة اسم الصورة إلى قائمة image_names
        image_names.append(image_path)
        
        # إحداثيات مربع التحديد مع المعلومات الأخرى (افتراضية)
        xmin, ymin, width, height = gt_rect
        xmax = xmin + width
        ymax = ymin + height
        label = [xmin, ymin, xmax, ymax, video_id, track_id, idx, 0, 0]  # استخدم 0 كـ class_id و occl
        labels.append(label)

# حفظ image_names.txt
with open('image_names.txt', 'w') as f:
    for name in image_names:
        f.write(name + '\n')

# حفظ labels.npy
labels_array = np.array(labels)
np.save('labels.npy', labels_array)