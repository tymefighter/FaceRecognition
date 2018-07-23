from PIL import Image
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import cv2
import keras
import pickle

#-----------------------------------------------------------------------

#Dataset

images_a = [cv2.imread(file) for file in glob.glob("Train/Ahmed/*")]
images_d = [cv2.imread(file) for file in glob.glob("Train/Devansh/*")]
images_n = [cv2.imread(file) for file in glob.glob("Train/Negative/*")]

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

#-----------------------------------------------------------------------

def create_placeholders(n_H0, n_W0, n_C0):
    X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0])
    #Y = tf.placeholder(tf.float32, [None, n_y])
    return X

def initialize_params():
	#Convolutional layers
	W1 = tf.get_variable("W1", [7,7,3,6], initializer=tf.contrib.layers.xavier_initializer(seed=0))
	W2 = tf.get_variable("W2", [7,7,6,8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
	W3 = tf.get_variable("W3", [7,7,8,14], initializer=tf.contrib.layers.xavier_initializer(seed=0))
	W4 = tf.get_variable("W4", [7,7,14,18], initializer=tf.contrib.layers.xavier_initializer(seed=0))
	W5 = tf.get_variable("W5", [7,7,18,24], initializer=tf.contrib.layers.xavier_initializer(seed=0))
	W6 = tf.get_variable("W6", [7,7,24,32], initializer=tf.contrib.layers.xavier_initializer(seed=0))
	
	#Fully Connected layers
	W7 = tf.get_variable("W7", [128,128], initializer=tf.contrib.layers.xavier_initializer(seed=0))
	W8 = tf.get_variable("W8", [128,128], initializer=tf.contrib.layers.xavier_initializer(seed=0))
	b7 = tf.get_variable("b7", [128,1], initializer=tf.zeros_initializer())
	b8 = tf.get_variable("b8", [128,1], initializer=tf.zeros_initializer())
	
	parameters = {"W1":W1, "W2":W2, "W3":W3, "W4":W4, "W5":W5, "W6":W6, "W7":W7, "W8":W8, "b7":b7, "b8":b8}
	return parameters
	
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

def compute_cost(A, P, N, alpha = 0.2):
	pos_dist = tf.reduce_sum((A-P)**2, axis=-1)
	neg_dist = tf.reduce_sum((A-N)**2, axis=-1)
	
	basic_loss = pos_dist - neg_dist + alpha
	loss = tf.reduce_sum(tf.maximum(basic_loss,0))

	return loss

def process_data(images, names):
	X_data = []
	x_data = [0,0,0]
	for i in range(8):
		for j in range(i+1,8):
			x_data[0] = images[i]
			x_data[1] = images[j]
			x_data[2] = images[i+16]
			X_data.append(x_data)
	for i in range(8,16):
		for j in range(i+1,16):
			x_data[0] = images[i]
			x_data[1] = images[j]
			x_data[2] = images[i+8]
			X_data.append(x_data)
	return X_data
	

def model(X_data, learning_rate = 0.00001, num_iterations=100, alpha=0.2, print_cost=True):
	

	X1 = create_placeholders(200, 200, 3)
	X2 = create_placeholders(200, 200, 3)
	X3 = create_placeholders(200, 200, 3)

	parameters = initialize_params()
	A = forward_propagation(X1, parameters)
	P = forward_propagation(X2, parameters)
	N = forward_propagation(X3, parameters)

	cost = compute_cost(A, P, N, alpha)

	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

	init = tf.global_variables_initializer()
	costs = []
	
	total_cost = 0
	with tf.Session() as sess:
		sess.run(init)
		for i in range(num_iterations):
			single_cost = 0
			total_cost = 0
			for j in range(len(X_data)):
				#print(len(X_data[j]))
				_, single_cost = sess.run([optimizer, cost], feed_dict = {X1: X_data[j][0], X2: X_data[j][1], X3: X_data[j][2]})
				total_cost += single_cost
		
			if(i%5 == 0 and print_cost == True):
				print("Total cost at " + str(i) + " iterations is " + str(total_cost))
			if(i%4 == 0):
				costs.append(total_cost)
		
		plt.plot(np.squeeze(costs))
		plt.ylabel('cost')
		plt.xlabel('iterations')
		plt.title("Learning rate =" + str(learning_rate))
		plt.show()
		parameter_values = sess.run(parameters)

	return parameter_values

#Main code

for i in range(len(images)):
	images[i] = np.reshape(images[i], [1,images[i].shape[0],images[i].shape[1],images[i].shape[2]])

X_data = process_data(images, names)
print(len(X_data))
parameters = model(X_data,0.000001,40,0.2,True)

with open('parameter_data.pickle', 'wb') as handle:
    pickle.dump(parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)
