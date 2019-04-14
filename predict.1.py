## TODO, write training code
import glob
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Test Run
from PIL import Image

# Create Model
from keras.models import load_model

import os

def save_matrix(a, filename):
    mat = np.matrix(a)
    with open(filename,'wb') as f:
        for line in mat:
            np.savetxt(f, line, fmt='%.2f')



import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
model = load_model('epoch1itmo.h5')


import tensorflow as tf
from keras import backend as K


# import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
# num_CPU = 1 
# num_GPU = 0

# config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
#                         inter_op_parallelism_threads=num_cores, 
#                         allow_soft_placement=True,
#                         device_count = {'CPU' : num_CPU,
#                                         'GPU' : num_GPU}
#                        )

# session = tf.Session(config=config)
# K.set_session(session)

## Grab images from the images_to_predict folder
image_list_input = []
for filename in glob.glob('images_to_predict/input/*.png'):
    image_list_input.append(filename)

# while (True): # Set infinite loop to allow for next epoch one all the images are used
for idx in range(0, len(image_list_input), 1):
    imagebatch_in = image_list_input[idx:idx + 1]
    print('Grabbing ', len(imagebatch_in), ' input files')
    YUV_list = []
    for img in imagebatch_in:
        openimg =Image.open(img)
        area = (128, 128, 384, 384)
        croppedimg = openimg.crop(area)
        # img_y, img_b, img_r = croppedimg.convert('YCbCr').split() # Obtain split, to extract Y channel
        # YUVArray = np.zeros((256,256,3), 'uint8')
        # YUVArray[..., 0] = np.true_divide(np.asarray(img_y).astype(float), 255)
        # YUVArray[..., 1] = np.true_divide(np.asarray(img_b).astype(float), 255)
        # YUVArray[..., 2] = np.true_divide(np.asarray(img_r).astype(float), 255)
        img_val = np.true_divide(np.asarray(croppedimg).astype(float), 255) # Obtain split, to extract Y channel
        YUV_list.append(img_val)
        X = np.asarray(YUV_list)
        pred = model.predict(X)
        print("prediction")
        imgpred = (pred * 255)[0].astype('uint8')
        
        print(imgpred, imgpred.shape)
        # YUVArrayout = (model.predict(X) * 255)[0,:,:,0].astype(int)
        # img_val = np.divide(np.asarray(img_y).astype(float), 255)
        # conv_img = img_val[:, :, np.newaxis] # Convert (512, 512) to (512, 512, 1)
        # YUV_list.append(conv_img)
        # X = np.asarray(YUV_list)

        # Recreate image
        # YUVArrayout = (model.predict(YUVArray[np.newaxis,:,:,:]) * 255)[0,:,:,0]
        # print(np.max(YUVArrayout), YUVArrayout.shape)
        # YUVArray = np.zeros((256,256,3), 'uint8')
        # YUVArray[..., 0] = np.asarray(YUVArrayout).astype(float)
        # YUVArray[..., 1] = np.asarray(img_b).astype(float)
        # YUVArray[..., 2] = np.asarray(img_r).astype(float)
        # print(YUVArrayout, YUVArrayout.shape)
        newimg = Image.fromarray(imgpred)
        # newimg = Image.fromarray(YUVArrayout, 'YCbCr').convert('RGB')
        # print(y_pred.shape, img_b.shape,img_r.shape)
        # newimg = Image.merge('YCbCr', [Image.fromarray(y_pred).convert('L'),img_b.convert('L'),img_r.convert('L')]).convert('RGB')
        # newimg.show()
        # save_matrix(YUVArrayout, 'images_to_predict/output/'+ img.split('\\')[-1]+'.txt')
        newimg.save('images_to_predict/output/'+ img.split('\\')[-1])

        openimg.close()
        



