## TODO, write training code

from model import *
from keras.preprocessing.image import ImageDataGenerator
# Test Run

print("Input Data")
trainX, testX = input_image()

print("Output Data")
trainY, testY = output_image()

# Begin Training
epochs = 100
batchsize = 2

# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest")

model = U_net()

# train the network
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=batchsize),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // batchsize,
	epochs=epochs)


# model = U_net()
# model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)

# data_gen_args = dict(rotation_range=0.2,
#                     width_shift_range=0.05,
#                     height_shift_range=0.05,
#                     shear_range=0.05,
#                     zoom_range=0.05,
#                     horizontal_flip=True,
#                     fill_mode='nearest')
# myGene = trainGenerator(2,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)

# model = unet()
# model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
# model.fit_generator(myGene,steps_per_epoch=300,epochs=1,callbacks=[model_checkpoint])

# testGene = testGenerator("data/membrane/test")
# results = model.predict_generator(testGene,30,verbose=1)
# saveResult("data/membrane/test",results)
