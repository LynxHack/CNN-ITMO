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
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
model = load_model('saved7-model-218-0.73.hdf5')


# # import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
# # num_CPU = 1 
# # num_GPU = 0

# # config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
# #                         inter_op_parallelism_threads=num_cores, 
# #                         allow_soft_placement=True,
# #                         device_count = {'CPU' : num_CPU,
# #                                         'GPU' : num_GPU}
# #                        )

# # session = tf.Session(config=config)
# # K.set_session(session)

# ## Grab images from the images_to_predict folder
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
        # area = (128, 128, 384, 384)
        # croppedimg = openimg.crop(area)
        
        img_val = np.true_divide(np.asarray(openimg).astype(float), 255) # Obtain split, to extract Y channel
        YUV_list.append(img_val)
        X = np.asarray(YUV_list)
        pred = model.predict(X)
        print("prediction")
        imgpred = (pred * 255)[0].astype('uint8')
        
        print(imgpred, imgpred.shape)

        newimg = Image.fromarray(imgpred)
        # save_matrix(YUVArrayout, 'images_to_predict/output/'+ img.split('\\')[-1]+'.txt')
        newimg.save('images_to_predict/output/'+ img.split('\\')[-1])

        openimg.close()
        



