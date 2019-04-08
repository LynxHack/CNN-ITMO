from model import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

itmo = trainGenerator(2,'data/train','input','output',data_gen_args,save_to_dir = None)

model = U_net()
model_checkpoint = ModelCheckpoint('unet_itmo.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(itmo, steps_per_epoch=300,epochs=1,callbacks=[model_checkpoint])

# testGene = testGenerator("data/membrane/test")
# results = model.predict_generator(testGene,30,verbose=1)
# saveResult("data/membrane/test",results)
