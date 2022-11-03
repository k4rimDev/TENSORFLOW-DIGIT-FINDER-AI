# make a prediction for a new image.
import glob
from numpy import argmax

from tensorflow.keras.utils import load_img
#from keras.preprocessing.image import load_img
from tensorflow.keras.utils import img_to_array
from keras.models import load_model

# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, grayscale=True, target_size=(28, 28))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 1 channel
	img = img.reshape(1, 28, 28, 1)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img

# load an image and predict the class
def run_example():
	for image in glob.glob("images/*.png"):
		load_images(image)

def load_images(image):
	# load the image
	img = load_image(image)
	# load model
	model = load_model('final_model.h5')
	# predict the class
	predict_value = model.predict(img)
	digit = argmax(predict_value)
	print(f"{image} --> {digit}")
# entry point, run the example
run_example()