import cv2 as cv
import glob
import numpy as np
imagepath = 'data/train/input/input'
imgs_names = glob.glob(imagepath+'/*.jpg')
for imgname in imgs_names:
    img = cv.imread(imgname)
    if img is None:
        print(imgname)

from PIL import Image
img = Image.open('data/train/input1/input/001.png')
img_yuv = img.convert('YCbCr')
matrix = np.asarray(img_yuv).astype(float)
print(matrix)

