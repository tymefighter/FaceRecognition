import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' 
import numpy as np
import tensorflow as tf
import cv2
import glob
import keras
from PIL import Image
import pickle

#-----------------------------------------

#Getting the parameters and the encodings

with open('parameter_data.pickle', 'rb') as handle:
    parameters = pickle.load(handle)

with open('encodings.pickle', 'rb') as handle:
	encodings = pickle.load(handle)

#Functions

def convert_to_tensor(parameters):
	tensor_params = {}
	tensor_params['W1'] = tf.convert_to_tensor(parameters['W1'],dtype=tf.float32)
	tensor_params['W2'] = tf.convert_to_tensor(parameters['W2'],dtype=tf.float32)
	tensor_params['W3'] = tf.convert_to_tensor(parameters['W3'],dtype=tf.float32)
	tensor_params['W4'] = tf.convert_to_tensor(parameters['W4'],dtype=tf.float32)
	tensor_params['W5'] = tf.convert_to_tensor(parameters['W5'],dtype=tf.float32)
	tensor_params['W6'] = tf.convert_to_tensor(parameters['W6'],dtype=tf.float32)
	tensor_params['W7'] = tf.convert_to_tensor(parameters['W7'],dtype=tf.float32)
	tensor_params['W8'] = tf.convert_to_tensor(parameters['W8'],dtype=tf.float32)
	tensor_params['b7'] = tf.convert_to_tensor(parameters['b7'],dtype=tf.float32)
	tensor_params['b8'] = tf.convert_to_tensor(parameters['b8'],dtype=tf.float32)
	return tensor_params

def forward_propagation(X, parameters):
	W1 = parameters['W1']
	W2 = parameters['W2']
	W3 = parameters['W3']
	W4 = parameters['W4']
	W5 = parameters['W5']
	W6 = parameters['W6']
	W7 = parameters['W7']
	W8 = parameters['W8']

	b7 = parameters['b7']
	b8 = parameters['b8']

	#CONV->BN->RELU->MAX_POOL
	Z1 = tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='VALID')
	B1 = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones')(Z1)
	A1 = tf.nn.relu(B1)
	P1 = tf.nn.max_pool(A1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

	#CONV->BN->RELU->MAX_POOL
	Z2 = tf.nn.conv2d(P1, W2, strides=[1,1,1,1], padding='VALID')
	B2 = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones')(Z2)
	A2 = tf.nn.relu(B2)
	P2 = tf.nn.max_pool(A2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

	#CONV->BN->RELU->CONV->BN->RELU->MAX_POOL
	Z3 = tf.nn.conv2d(P2, W3, strides=[1,1,1,1], padding='VALID')
	B3 = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones')(Z3)
	A3 = tf.nn.relu(B3)
	Z4 = tf.nn.conv2d(A3, W4, strides=[1,1,1,1], padding='VALID')
	B4 = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones')(Z4)
	A4 = tf.nn.relu(B4)
	P4 = tf.nn.max_pool(A4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

	#CONV->BN->RELU->CONV->BN->RELU->MAX_POOL
	Z5 = tf.nn.conv2d(P4, W5, strides=[1,1,1,1], padding='VALID')
	B5 = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=100, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones')(Z5)
	A5 = tf.nn.relu(B5)
	Z6 = tf.nn.conv2d(A5, W6, strides=[1,1,1,1], padding='VALID')
	B6 = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=100, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones')(Z6)
	A6 = tf.nn.relu(B6)
	P6 = tf.nn.max_pool(A6, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

	#Flattening out P6
	P6 = tf.contrib.layers.flatten(P6)
	P6 = tf.transpose(P6)
	#FC->FC
	Z7 = tf.matmul(W7,P6) + b7
	A7 = tf.nn.relu(Z7)
	
	Z8 = tf.matmul(W8, A7) + b8

	return Z8

def check(X, other_encoding, parameters):	#X must be a numpy array and parametes must be a dictionary containing numpy arrays

	X = tf.convert_to_tensor(X, tf.float32)	#converting to a tensor
	parameters = convert_to_tensor(parameters)	#Converting to a dictionary containing tensors
	
	encoding_tensor = forward_propagation(X, parameters)
	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		encoding = sess.run(encoding_tensor)

	dist = np.sum((other_encoding - encoding)**2, axis = 0)
	return dist

def predict(X, parameters, encodings, names_encodings, tau=0.1):
	#X->numpy array which would hold the image
	#parameters->Dictionary which holds the parameters in ndarray form
	#Encodings is a list of encodings(numpy arrays) of each person
	#names->names of the encodings
	least = float("inf")
	for i in range(len(encodings)):
		dist = check(X, encodings[i], parameters)
		if(dist < least):
			least = dist
			name = names_encodings[i]
	if(least > tau):
		return "Negative"
	return name 

#Loading the test data

images_a = [cv2.imread(file) for file in glob.glob("Test/Ahmed/*")]
images_d = [cv2.imread(file) for file in glob.glob("Test/Devansh/*")]
images_n = [cv2.imread(file) for file in glob.glob("Test/Negative/*")]

names_a = []
names_d = []
names_n = []

for i in range(len(images_a)):
    images_a[i] = cv2.resize(images_a[i], (200, 200))
    names_a.append('Ahmed')
for i in range(len(images_d)):
    images_d[i] = cv2.resize(images_d[i], (200, 200))
    names_d.append('Devansh')
for i in range(len(images_n)):
    images_n[i] = cv2.resize(images_n[i], (200, 200))
    names_n.append('Negative')

images = []
images = images_a + images_d + images_n
names = names_a + names_d + names_n
images_reshaped = images

for i in range(len(images)):
	images_reshaped[i] = np.reshape(images[i], [1,images[i].shape[0],images[i].shape[1],images[i].shape[2]])

#Loading the training data

images_a_t = [cv2.imread(file) for file in glob.glob("Train/Ahmed/*")]
images_d_t = [cv2.imread(file) for file in glob.glob("Train/Devansh/*")]
images_n_t = [cv2.imread(file) for file in glob.glob("Train/Negative/*")]

names_a_t = []
names_d_t = []
names_n_t = []

for i in range(len(images_a_t)):
    images_a_t[i] = cv2.resize(images_a_t[i], (200, 200))
    names_a_t.append('Ahmed')
for i in range(len(images_d_t)):
    images_d_t[i] = cv2.resize(images_d_t[i], (200, 200))
    names_d_t.append('Devansh')
for i in range(len(images_n_t)):
    images_n_t[i] = cv2.resize(images_n_t[i], (200, 200))
    names_n_t.append('Negative')

names_encodings = names_a_t + names_d_t

#Main code


for i in range(len(images_reshaped)):
	name = predict(images_reshaped[i], parameters, encodings, names_encodings, 0.1)
	print(images[i].shape)
	cv2.imshow(name, np.reshape(images_reshaped[i],[images_reshaped[i].shape[1],images_reshaped[i].shape[2],images_reshaped[i].shape[3]]))
	cv2.waitKey(4000)
	cv2.destroyAllWindows()
