# CNN-ITMO
Using CNN to convert from SDR to HDR

## Packages to install
1. Pillow
2. Keras
3. Tensorflow
4. Numpy

## GPU options:
Install tensorflow for GPU

To see if GPU is working insert lines

~~~~
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
~~~~

## Running the file

~~~~
python3 main.py
~~~~
