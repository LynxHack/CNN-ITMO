## TODO, write training code

from model import *
# from datagen import *
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger
# Test Run
from PIL import Image

# Create Model
import keras
from keras.models import load_model
from keras.losses import mean_squared_error
import signal
import sys

# Test Run

import os

import glob
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import skimage.io as io
import skimage.transform as trans

import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)

# try:
# 	model = load_model('itmo.h5') #continue training saved model weight weights
# 	print("successfully loaded model from previous save file")
# except:
# 	print('model not found, creating a new model')
# 	model = U_net()

model = load_model("saved6-model-136-0.77.hdf5")

# model.save('itmo.h5')  # creates a HDF5 file 'my_model.h5'
def makePrediction(epoch, logs) :
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
			newimg.save('images_to_predict/output/'+ 'epochZZZ'+str(epoch)+img.split('\\')[-1])

			openimg.close()
testmodelcb = keras.callbacks.LambdaCallback(on_epoch_end=makePrediction)
# cust_train_generator = image_gen(inputfile='data/train/input1/input/*.png', 
# 							outputfile='data/train/output1/output/*.png',
# 							n_chunks=2) ## save once per train batch in case of closing halfway

# cust_test_generator = validation_image_gen(inputfile='data/test/input1/input/*.png', 
# 							outputfile='data/test/output1/output/*.png',
# 							n_chunks=1)

# while(True):
# X2, y2 = next(cust_train_generator)

# X, y = next(cust_test_generator)

# test_generator = zip(testimage_generator, testmask_generator)
# csv_logger = CSVLogger('log.csv', append=True, separator=';')

# index = 0
# while True:
# 	while index <= 1000:
# 		X2, y2 = next(cust_train_generator)
# 		X, y = next(cust_test_generator)
# 		train_history = model.fit(X2, y2, epochs = 1, verbose=1, validation_data = (X2,y2), callbacks=[csv_logger])
# 		index = index + 1
# 		print(index)
# 	index = 0
# 	model.save("epoch"+str(index)+".hdf5")

data_gen_args = dict(rescale=1. / 255,
					rotation_range = 90,
					horizontal_flip = True,
					vertical_flip=True,
					zoom_range=0.2,
)

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

seed = 1
image_generator = image_datagen.flow_from_directory(
    'data/train/input1',
	target_size=(512,512),
	color_mode='rgb',
    class_mode=None,
	batch_size=2,
	shuffle=True,
    seed=seed)

mask_generator = mask_datagen.flow_from_directory(
    'data/train/output1',
	target_size=(512,512),
	color_mode='rgb',
    class_mode=None,
	batch_size=2,
	shuffle=True,
    seed=seed)
	
train_generator = zip(image_generator, mask_generator)

testimage_generator = image_datagen.flow_from_directory(
    'data/test/input1',
	target_size=(512,512),
	color_mode='rgb',
    class_mode=None,
	batch_size=1,
	shuffle=True,
    seed=seed)

testmask_generator = mask_datagen.flow_from_directory(
    'data/test/output1',
	target_size=(512,512),
	color_mode='rgb',
    class_mode=None,
	batch_size=1,
	shuffle=True,
    seed=seed)

test_generator = zip(testimage_generator, testmask_generator)
csv_logger = CSVLogger('log.csv', append=True, separator=';')

filepath = "saved7-model-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=False, mode='max')

model.fit_generator(generator=train_generator,
					validation_data=test_generator,
					validation_steps = 100,
					steps_per_epoch=1000,
					epochs=1000,
					verbose=1,
					callbacks=[csv_logger, checkpoint, testmodelcb])
