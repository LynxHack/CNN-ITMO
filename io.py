import Image
import numpy as np

im = np.array(Image.open(src)).astype('float32')/255

print(im.shape)