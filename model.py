# FILE IO
from PIL import Image
import glob
import matplotlib.pyplot as plt

# Scikit
import numpy as np 
from numpy import array
import os
# import skimage.io as io
# import skimage.transform as trans

import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.preprocessing import image
# import BatchNormalization
from keras.layers.normalization import BatchNormalization

## YUV Extraction from RGB
#import cv2 

# def input_image():
#     image_list = []

#     for filename in glob.glob('InputData/*.tiff'): #assuming tiff
#         im=Image.open(filename)
#         image_list.append(im)

#     ## Convert into YUV then extract luma info, append into training and test set data array for one epoch
#     YUV_list = []
#     for img in image_list:
#         img_y, img_b, img_r = img.convert('YCbCr').split()
#         img_y_np = [np.asarray(img_y).astype(float) // 255.0]
#         YUV_list.append(img_y_np)
#         # print(img_y_np)


#     ## Split to training and test sets
#     n = len(YUV_list)
#     ratio = 1
#     train_X = np.asarray(YUV_list[0 : round(ratio * n)])
#     test_X = np.asarray(YUV_list[round(ratio * n) + 1: n - 1])

#     return train_X.reshape(512,512,1), test_X.reshape(512,512,1)

def image_gen(inputfile, outputfile, n_chunks, model):
    image_list_input = []
    image_list_output = []
    for filename in glob.glob(inputfile):
        # im=Image.open(filename)
        # image_list_input.append(im)
        image_list_input.append(filename)

    for filename in glob.glob(outputfile):
        # im=Image.open(filename)
        # image_list_output.append(im)
        image_list_output.append(filename)
        
    ## Convert into YUV append into X and y set data array for one epoch
    print('generator initiated')
    while (True): # Set infinite loop to allow for next epoch one all the images are used
        for idx in range(0, len(image_list_input), n_chunks):
            imagebatch_in = image_list_input[idx:idx + n_chunks]
            imagebatch_out = image_list_output[idx:idx + n_chunks]
            # print(imagebatch_in)
            # print(imagebatch_out)
            print('Grabbing ', len(imagebatch_in), ' input files')
            print('Grabbing ', len(imagebatch_out), ' output files')
            YUV_list = []
            for img in imagebatch_in:
                openimg =Image.open(img)
                img_y, img_b, img_r = openimg.convert('YCbCr').split() # Obtain split, to extract Y channel
                img_val = np.asarray(img_y).astype(float) // 255
                conv_img = img_val[:, :, np.newaxis] # Convert (512, 512) to (512, 512, 1)
                YUV_list.append(conv_img)
                X = np.asarray(YUV_list)
                openimg.close()

            YUV_list = []
            for img in imagebatch_out: # Do the same for output images
                openimg =Image.open(img)
                img_y, img_b, img_r = openimg.convert('YCbCr').split() # Obtain split, to extract Y channel
                img_val = np.asarray(img_y).astype(float) // 255
                conv_img = img_val[:, :, np.newaxis] # Convert (512, 512) to (512, 512, 1)
                YUV_list.append(conv_img)
                y = np.asarray(YUV_list)
                openimg.close()

            yield X, y
            model.save('itmo.h5')
            print('generator yielded a batch starting from image #%d' % idx)



# def image_gen(inputfile, outputfile, n_chunks):
#     image_list_input = []
#     image_list_output = []
#     for filename in glob.glob(inputfile):
#         # im=Image.open(filename)
#         # image_list_input.append(im)
#         image_list_input.append(filename)

#     for filename in glob.glob(outputfile):
#         # im=Image.open(filename)
#         # image_list_output.append(im)
#         image_list_output.append(filename)
        
#     ## Convert into YUV append into X and y set data array for one epoch
#     print('generator initiated')
#     while (True): # Set infinite loop to allow for next epoch one all the images are used
#         for idx in range(0, len(image_list_input), n_chunks): 
#             imagebatch_in = image_list_input[idx:idx + n_chunks]
#             imagebatch_out = image_list_output[idx:idx + n_chunks]
#             # print(imagebatch_in)
#             # print(imagebatch_out)
#             print('Grabbing ', len(imagebatch_in), ' input files')
#             print('Grabbing ', len(imagebatch_out), ' output files')
#             YUV_list = []
#             for img in imagebatch_in:
#                 openimg =Image.open(img)
#                 img_val = np.asarray(openimg.convert('YCbCr')).astype(float)
#                 YUV_list.append(img_val)
#                 X = np.asarray(YUV_list)
#                 openimg.close()

#             YUV_list = []
#             for img in imagebatch_out:
#                 openimg =Image.open(img)
#                 img_val = np.asarray(openimg.convert('YCbCr')).astype(float)
#                 YUV_list.append(img_val)
#                 y = np.asarray(YUV_list) 
#                 openimg.close()

#             yield X, y

#             print('generator yielded a batch starting from image #%d' % idx)




def validation_image_gen(inputfile, outputfile, n_chunks):
    image_list_input = []
    image_list_output = []
    for filename in glob.glob(inputfile):
        # im=Image.open(filename)
        # image_list_input.append(im)
        image_list_input.append(filename)

    for filename in glob.glob(outputfile):
        # im=Image.open(filename)
        # image_list_output.append(im)
        image_list_output.append(filename)

    ## Convert into YUV append into X and y set data array for one epoch
    print('generator initiated')
    while True:
        imagebatch_in = image_list_input
        imagebatch_out = image_list_output
        print('Grabbing ', len(imagebatch_in), 'validation files')
        YUV_list = []
        for img in imagebatch_in:
            openimg =Image.open(img)
            img_y, img_b, img_r = openimg.convert('YCbCr').split() # Obtain split, to extract Y channel
            img_val = np.asarray(img_y).astype(float) // 255
            conv_img = img_val[:, :, np.newaxis] # Convert (512, 512) to (512, 512, 1)
            YUV_list.append(conv_img)
            X = np.asarray(YUV_list)
            openimg.close()

        YUV_list = []
        for img in imagebatch_out:
            openimg =Image.open(img)
            img_y, img_b, img_r = openimg.convert('YCbCr').split() # Obtain split, to extract Y channel
            img_val = np.asarray(img_y).astype(float) // 255
            conv_img = img_val[:, :, np.newaxis] # Convert (512, 512) to (512, 512, 1)
            YUV_list.append(conv_img)
            y = np.asarray(YUV_list)
            openimg.close()

        yield X, y


        print('generator yielded a batch starting from image #%d' % idx)

def load_image(filename, ratio):
    image_list = []

    for filename in glob.glob(filename): #assuming tiff
        im=Image.open(filename)
        image_list.append(im)

    print(len(image_list),': number of images')

    ## Convert into YUV then extract luma info, append into training and test set data array for one epoch
    YUV_list = []
    for img in image_list:
        # img_y, img_b, img_r = img.convert('YCbCr').split()
        img_val = np.asarray(img.convert('YCbCr')).astype(float) // 255.0
        # img_val = np.asarray([np.asarray(img_y).astype(float), 
        #                       np.asarray(img_b).astype(float), 
        #                       np.asarray(img_r).astype(float)]) //255.0
        YUV_list.append(img_val)
        # print(img_y_np)

    ## Split to training and test sets outputs
    n = len(YUV_list)
    train = np.asarray(YUV_list[0 : round(ratio * n)])
    test = np.asarray(YUV_list[round(ratio * n) + 1: n - 1]) 

    return train, test

####IMPORTANT(4.10)
## Add Batch After Relu
def ConvBN(filters, kernel_size, inputs):
    return BatchNormalization()(Activation(activation='relu')(Conv2D(filters, kernel_size, padding = 'same', kernel_initializer = 'he_normal')(inputs)))

## 
def ConvBNTranspose(filters, kernel_size, inputs):    
    return BatchNormalization()(Activation(activation='relu')(Conv2DTranspose(filters, kernel_size, strides=2, padding = 'valid', kernel_initializer = 'he_normal')(inputs)))
    
    ##Modified to use only Y component (4.10)
def U_net(pretrained_weights = None, input_size = (512,512,1)):
    ##Encoding
    ##32 kernels for the first block with size 3*3  
    inputs = Input(input_size)
    conv1 = ConvBN(32, 3, inputs)
    conv1 = ConvBN(32, 3, conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2),strides=2)(conv1)

    ##64 kernels for the second block with size 3*3
    conv2 = ConvBN(64, 3, pool1)
    conv2 = ConvBN(64, 3, conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2),strides=2)(conv2)

    ##128 kernels for the third block with size 3*3
    conv3 = ConvBN(128, 3, pool2)
    conv3 = ConvBN(128, 3, conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2),strides=2)(conv3)
    
    ##256 kernels for the fourth block with size 3*3
    ##Delete Dropout here. Use data augmentation to solve the overfitting
    conv4 = ConvBN(256, 3, pool3)
    conv4 = ConvBN(256, 3, conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2),strides=2)(drop4)

    ##512 kernels for the fifth block
    conv5 = ConvBN(512, 3, pool4)
    conv5 = ConvBN(512, 3, conv5)
    drop5 = Dropout(0.5)(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2),strides=2)(drop5)

    ##1024 kernel for the 6th block. Pass to decoder
    ##Dropout here(4.10)
    conv_cross = ConvBN(1024, 3, pool5)
    conv_cross = ConvBN(1024, 3, conv_cross)
    drop_cross = Dropout(0.5)(conv_cross)

    ##Decoding 
    ##transposed conv
    ##upsample fiter size 4 (2*2),strides (2) 
    ##concatenate on axis 3
    up6 = ConvBNTranspose(512, 2, drop_cross)##(UpSampling2D(size = (2,2))(conv_cross))
    merge6 = concatenate([drop5, up6], axis = 3)
    conv6 = ConvBN(512, 3, merge6)
    conv6 = ConvBN(512, 3, conv6)
                                                                                       
    up7 = ConvBNTranspose(256, 2, conv6)##(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([drop4,up7], axis = 3)
    conv7 = ConvBN(256, 3,merge7)
    conv7 = ConvBN(256, 3,conv7)

    up8 = ConvBNTranspose(128, 2, conv7)##(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv3,up8], axis = 3)
    conv8 = ConvBN(128, 3, merge8)
    conv8 = ConvBN(128, 3, conv8)

    up9 = ConvBNTranspose(64, 2, conv8)##(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv2,up9], axis = 3)
    conv9 = ConvBN(64, 3, merge9)
    conv9 = ConvBN(64, 3, conv9)
    # conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    
    up10 = ConvBNTranspose(32, 2, conv9)##(UpSampling2D(size = (2,2))(conv9))
    merge10 = concatenate([conv1,up10], axis = 3)
    conv10 = ConvBN(32, 3, merge10)
    conv10 = ConvBN(32, 3, conv10)
    #conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9) 
    
    ## To generate output, use 3 filters with size of 3*3      
    ## Use Sigmoid here (changed by zz)      
    ## One dimension in Z axis, and 3*3 filter size
    ## Second parameter 1 or 3
    OutImage = Conv2D(1, 1, activation = 'sigmoid')(conv10)

    model = Model(input = inputs, output = OutImage, name='Reinhardt Prediction')
    # Adam Optimizer
    # adam = optimizers.Adam(lr=0.002, beta_1=0.5, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer = Adam(lr=0.002, beta_1=0.5, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), loss = 'mean_squared_error', metrics = ['accuracy'])
    # model.compile(optimizer = SGD(lr=0.01, momentum=0.09, decay=1e-6, nesterov=True), loss = 'mean_squared_error', metrics = ['accuracy'])
#   Calculate the mean square error
#     if(pretrained_weights):
#     	model.load_weights(pretrained_weights)

    model.summary()
    return model
#     model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    



#     return model