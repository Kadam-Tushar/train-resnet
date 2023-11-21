import os 
import random 
import shutil

image_net_folder = './data/train'
for (root,dir,files) in os.walk(image_net_folder):
    if len(files) != 600:
        continue

    test_files = [os.path.join(root,x) for x in  random.sample(files,100)]
    test_folder = os.path.join("./data/test",root.split("/")[-1])
    os.mkdir(test_folder)
    for x in test_files:
        shutil.move(x,os.path.join(test_folder,x.split("/")[-1]))
