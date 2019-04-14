import cv2 as cv
import glob

imagepath = 'data/train/input/input'
imgs_names = glob.glob(imagepath+'/*.jpg')
for imgname in imgs_names:
    img = cv.imread(imgname)
    if img is None:
        print(imgname)