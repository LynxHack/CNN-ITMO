# CNN-ITMO
Using CNN to convert from SDR to HDR

## Installing Python
Please install python3.X, either directly, or through Anaconda

## PIP Packages to install (dependencies)
1. Pillow
2. Keras
3. Tensorflow-gpu (please confirm the corresponding version compatibility with both keras and CUDA)
4. Numpy

### How to install pip packages examples
~~~~
pip install Pillow
pip install Numpy
~~~~

## GPU options (optional for prediction, mandatory for training):
1) Install tensorflow for GPU (Tensorflow-gpu)
2) Install CUDA and Cudnn (ensure they are version compatible according to Nvidia website)
*Please also ensure that the Tensorflow-gpu version is compatible with Keras version and also the CUDA version.


To see if GPU is working, run this inside python CLI

~~~~
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
~~~~

## Running the file

~~~~
python3 main.py
~~~~

## Inserting Training Set (For training use only)
Due to the large size of the training dataset, the images are not included in the repository.
Please insert the input images from the SDR Virtual Camera to the directory
~~~~
./data/train/input1/input/
~~~~

Correspondingly, the output images should go to
~~~~
./data/train/output1/output/
~~~~


## Running Prediction
Please insert the images that you would like to predict inside the directory

~~~~
./images_to_predict/input/
~~~~

Then run in the terminal (Linux / MacOS) or Powershell (Windows)

~~~~
python predict.py
~~~~

The predicted images would be populated in the corresponding output directory to be retrieved

# MATLAB Scripts

## Generating Input and Output Images
Please use virtual_camera.m for generating input images and Reinhard.m for generating output images (replace input images)

## Generating HDR images from the CNN output
Please run the inverse_Reinhard.m on the produced CNN output to generate the final HDR image (replace input images)