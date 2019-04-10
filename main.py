## TODO, write training code

from model import *
# from datagen import *
from keras.preprocessing.image import ImageDataGenerator
# Test Run
from PIL import Image
# 'GroundTruth/*.tiff'
# print("Input Data")
# trainX, testX = load_image('data/input/*.png', 1.0)
# print("Output Data")
# trainY, testY = load_image('data/output/*.png', 1.0)

# Begin Training
# epochs = 100
# batchsize = 32

# Create Model
model = U_net()

# Load Model
# model = load_model('itmo.h5')


train_generator = image_gen(inputfile='data/train/input/*.png', 
							outputfile='data/train/output/*.png',
							n_chunks=70)

test_generator = validation_image_gen(inputfile='data/test/input/*.png', 
							outputfile='data/test/output/*.png',
							n_chunks=70)

# X, y = next(train_generator)
# model.fit(X, y, validation_split=0.2, verbose = 1)

model.fit_generator(generator=train_generator,
					validation_data=test_generator,
					validation_steps = 10,
					steps_per_epoch=10, 
					epochs=1,
					verbose=1)
model.save('itmo.h5')  # creates a HDF5 file 'my_model.h5'

# def myFunc(image):
#     img = np.array(image)
#     imgYCC = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
#     return Image.fromarray(imgYCC)

# # construct the training image generator for batch loading
# datagen_args = dict(rescale=1. / 255, 
# 					rotation_range=180,
# 					# color_mode='rgb',
# 					horizontal_flip=True)
# 					# preprocessing_function = myFunc)

# datagenx = ImageDataGenerator(**datagen_args)
# datageny = ImageDataGenerator(**datagen_args)


# # datagenx.fit(training_images, augment=True, seed=1)
# # datageny.fit(training_masks, augment=True, seed=1)


# genx= datagenx.flow_from_directory('data/traininput',
# 								   target_size=(512,512),
# 								   class_mode=None,
# 								   seed=1)
# geny= datageny.flow_from_directory('data/trainoutput',
# 								   target_size=(512,512),
# 								   class_mode=None,
# 								   seed=1)

# # # you can now fit a generator as well
# # datagenX.fit_generator(dgdx, nb_iter=100)

# # here we sychronize two generator and combine it into one
# train_generator = zip(genx, geny)
