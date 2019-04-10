## TODO, write training code

from model import *
# from datagen import *
from keras.preprocessing.image import ImageDataGenerator
# Test Run
from PIL import Image

# Create Model
from keras.models import load_model

try:
	model = load_model('itmo.h5') #continue training saved model weight weights
	print("successfully loaded model from previous save file")
except:
	print('model not found, creating a new model')
	model = U_net()

train_generator = image_gen(inputfile='data/train/input/*.png', 
							outputfile='data/train/output/*.png',
							n_chunks=8, model = model) ## save once per train batch in case of closing halfway

test_generator = validation_image_gen(inputfile='data/test/input/*.png', 
							outputfile='data/test/output/*.png',
							n_chunks=139)

model.fit_generator(generator=train_generator,
					validation_data=test_generator,
					validation_steps = 1,
					steps_per_epoch=300, 
					epochs=1,
					verbose=1)

model.save('itmo.h5')  # creates a HDF5 file 'my_model.h5'
