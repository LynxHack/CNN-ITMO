from PIL import Image
import numpy as np
import glob
image_list = []

for filename in glob.glob('data/*.tiff'): #assuming tiff
    im=Image.open(filename)
    image_list.append(im)

for i in image_list:
    print(i)