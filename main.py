## TODO, write training code

from model import *
# from datagen import *
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger
# Test Run
from PIL import Image

# Create Model
from keras.models import load_model

import signal
import sys
def signal_handler(sig, frame):
        print('You pressed Ctrl+C!')
        model.save('itmo.h5')
        sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# try:
# 	model = load_model('itmo.h5') #continue training saved model weight weights
# 	print("successfully loaded model from previous save file")
# except:
# 	print('model not found, creating a new model')
# 	model = U_net()
model = U_net()
# train_generator = image_gen(inputfile='data/train/input/*.png', 
# 							outputfile='data/train/output/*.png',
# 							n_chunks=2,model = model) ## save once per train batch in case of closing halfway

# test_generator = validation_image_gen(inputfile='data/test/input/*.png', 
# 							outputfile='data/test/output/*.png',
# 							n_chunks=1)

data_gen_args = dict(featurewise_center=True,
					 rescale=1. / 255,
                     rotation_range=90,
					 horizontal_flip=True,
					 vertical_flip=True,
                     zoom_range=0.2)

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

seed = 1
image_generator = image_datagen.flow_from_directory(
    'data/train/input1',
	target_size=(512,512),
	color_mode='rgb',
    class_mode=None,
	batch_size=1,
	shuffle=True,
    seed=seed)

mask_generator = mask_datagen.flow_from_directory(
    'data/train/output1',
	target_size=(512,512),
	color_mode='rgb',
    class_mode=None,
	batch_size=1,
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

filepath = "saved-model-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')

model.fit_generator(generator=train_generator,
					validation_data=test_generator,
					validation_steps = 100,
					steps_per_epoch=2400,
					epochs=20,
					verbose=1,
					callbacks=[csv_logger, checkpoint])

model.save('itmo.h5')  # creates a HDF5 file 'my_model.h5'
