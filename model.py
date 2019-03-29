# FILE IO
from PIL import Image
import glob
import matplotlib.pyplot as plt

# Scikit
import numpy as np 
import os
# import skimage.io as io
# import skimage.transform as trans

# Import Keras
# from keras.models import *
# from keras.layers import *
# from keras.optimizers import *
# from keras.callbacks import ModelCheckpoint, LearningRateScheduler
# from keras import backend as keras
# from keras.preprocessing import image

## YUV Extraction from RGB
#import cv2 

def input_image():
    image_list = []

    for filename in glob.glob('InputData/*.tiff'): #assuming tiff
        im=Image.open(filename)
        image_list.append(im)

    ## Convert into YUV then extract luma info, append into training and test set data array for one epoch
    YUV_list = []
    for img in image_list:
        img_y, img_b, img_r = img.convert('YCbCr').split()
        img_y_np = np.asarray(img_y).astype(float)
        YUV_list.append(img_y_np)
        print(img_y_np)


    ## Split to training and test sets
    n = len(YUV_list)
    ratio = 0.8
    train_X = YUV_list[0 : round(ratio * n)]
    test_X = YUV_list[round(ratio * n) + 1: n - 1]

    return train_X, test_X

def output_image():
    image_list = []

    for filename in glob.glob('GroundTruth/*.tiff'): #assuming tiff
        im=Image.open(filename)
        image_list.append(im)

    ## Convert into YUV then extract luma info, append into training and test set data array for one epoch
    YUV_list = []
    for img in image_list:
        img_y, img_b, img_r = img.convert('YCbCr').split()
        img_y_np = np.asarray(img_y).astype(float)
        YUV_list.append(img_y_np)
        print(img_y_np)

    ## Split to training and test sets outputs
    n = len(YUV_list)
    ratio = 0.8
    train_Y = YUV_list[0 : round(ratio * n)]
    test_Y = YUV_list[round(ratio * n) + 1: n - 1]

    return train_Y, test_Y

    ##The depth is changed to 3 for color image
def U_net(pretrained_weights = None,input_size = (512,512,3)):
    ##Encoding
    ##32 kernels for the first block with size 3*3  
    inputs = Input(input_size)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2),strides=2)(conv1)

    ##64 kernels for the second block with size 3*3
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2),strides=2)(conv2)

    ##128 kernels for the third block with size 3*3
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2),strides=2)(conv3)
    
    ##256 kernels for the fourth block with size 3*3
    ##Delete Dropout here. Use data augmentation to solve the overfitting
    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
#   drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2),strides=2)(conv4)

    ##512 kernels for the fifth block
     conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
     conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
#    drop5 = Dropout(0.5)(conv5)
     pool5 = MaxPooling2D(pool_size=(2, 2),strides=2)(conv5)

    ##1024 kernel for the 6th block. Pass to decoder
     conv_cross = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool5)
     conv_cross = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv_cross)

    ##Decoding 
    ##transposed conv
    ##upsample fiter size 4 (2*2),strides (2) 
    up6 = Conv2D(512, 2, activation = 'relu', strides=2, padding = 'valid', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv_cross
    merge6 = concatenate([conv5, up6], axis =3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
                                                                                       
    up7 = Conv2D(256, 2, activation = 'relu', strides=2; padding = 'valid', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv4,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', strides=2; padding = 'valid', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv3,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', strides=2; padding = 'valid', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv2,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    
    up10 = Conv2D(32, 2, activation = 'relu', strides=2; padding = 'valid', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv9))
    merge10 = concatenate([conv1,up10], axis = 3)
    conv10 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge10)
    conv10 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv10)
    #conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9) 
    
    ## To generate output, use 3 filters with size of 3*3      
    ## Use Softmax here (changed by zz)                                                                                                                                     
    OutImage = Conv2D(3, 3, activation = 'softmax')(conv10)

    model = Model(input = inputs, output = OutImage)

#     model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
#     #model.summary()

#     if(pretrained_weights):
#     	model.load_weights(pretrained_weights)

#     return model
