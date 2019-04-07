import tensorflow as tf
import datetime
import keras.backend as K
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import numpy as np
import pdb
import os
import cv2
from keras.applications import vgg19, VGG16
import utils
import loss_util
from keras.layers import Input
import time
from scipy.optimize import fmin_l_bfgs_b
from keras.preprocessing.image import load_img, save_img, img_to_array


# copy data
data_dir = 'data/'
file_index = 1

style_img_files = os.listdir(data_dir + 'style/')
indices = []
for file in style_img_files:
	file = file.split('_')[0]
	indices.append(file)

style_imgs = []
mask_imgs = []
mask_dilated_imgs = []
naive_imgs = []
for i in indices:
	style_img = cv2.imread(data_dir + 'style/' + i + '_target.jpg')
	style_imgs.append(style_img)
	mask_img = cv2.imread(data_dir + 'mask/' + i + '_c_mask.jpg')
	mask_imgs.append(mask_img)
	mask_dilated_img = cv2.imread(data_dir + 'mask_dilated/' + i + '_c_mask_dilated.jpg')
	mask_dilated_imgs.append(mask_dilated_img)
	naive_img = cv2.imread(data_dir + 'fusion/' + i + '_naive.jpg')
	naive_imgs.append(naive_img)

style_imgs = np.array(style_imgs)
mask_imgs = np.array(mask_imgs)
mask_dilated_imgs = np.array(mask_dilated_imgs)
naive_imgs = np.array(naive_imgs)



# particular case - 

style_img = style_imgs[file_index]
naive_img_o = naive_imgs[file_index]
mask_img = mask_imgs[file_index]
mask_dilated_img = mask_dilated_imgs[file_index]

# object_img = np.random.rand(500,300,3)
# style_img = np.random.rand(500,300,3)
# naive_img_o = np.random.rand(500,300,3)
# mask_img = np.random.rand(500,300,3)
# mask_dilated_img = utils.dilate_mask(mask_img)

# get tensor representations of our images

naive_img = K.variable(utils.preprocess_image(naive_img_o))
style_img = K.variable(utils.preprocess_image(style_img))
img_rows, img_cols = naive_img.shape[1] , naive_img.shape[2]

fusion_img = K.placeholder((1, img_rows, img_cols, 3))

# combine the 3 images into a single Keras tensor
input_tensor = K.concatenate([naive_img,style_img, fusion_img], axis=0)

# build the vgg16 network with our 3 images as input
# the model will be loaded with pre-trained ImageNet weights
model = VGG16(input_tensor=input_tensor,
					weights='imagenet', include_top=False)
print('Model loaded.')
# get the symbolic outputs of each "key" layer (we gave them unique names).
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

c_weight = 0.4
s_weight = 0.4
t_weight = 0.2

# combine these loss functions into a single scalar
loss = K.variable(0.0)
layer_features = outputs_dict['block5_conv2']
naive_img_content_features = layer_features[0, :, :, :]
fusion_img_content_features = layer_features[2, :, :, :]
loss += c_weight * loss_util.content_loss(naive_img_content_features,
										fusion_img_content_features)

feature_layers = ['block1_conv1', 'block2_conv1',
				  'block3_conv1', 'block4_conv1',
				  'block5_conv1']
for layer_name in feature_layers:
	layer_features = outputs_dict[layer_name]
	style_img_style_features = layer_features[1, :, :, :]
	fusion_img_style_features = layer_features[2, :, :, :]
	sl = loss_util.style_loss(fusion_img_style_features, style_img_style_features, img_rows, img_cols)
	loss += (s_weight / len(feature_layers)) * sl
loss += t_weight * loss_util.total_variation_loss(fusion_img, img_rows, img_cols)

# get the gradients of the generated image wrt the loss
grads = K.gradients(loss, fusion_img)

outputs = [loss]
if isinstance(grads, (list, tuple)):
	outputs += grads
else:
	outputs.append(grads)

f_outputs = K.function([fusion_img], outputs)


def eval_loss_and_grads(x):
	if K.image_data_format() == 'channels_first':
		x = x.reshape((1, 3, img_rows, img_cols))
	else:
		x = x.reshape((1, img_rows, img_cols, 3))
	outs = f_outputs([x])
	loss_value = outs[0]
	if len(outs[1:]) == 1:
		grad_values = outs[1].flatten().astype('float64')
	else:
		grad_values = np.array(outs[1:]).flatten().astype('float64')
	return loss_value, grad_values

# this Evaluator class makes it possible
# to compute loss and gradients in one pass
# while retrieving them via two separate functions,
# "loss" and "grads". This is done because scipy.optimize
# requires separate functions for loss and gradients,
# but computing them separately would be inefficient.


class Evaluator(object):

	def __init__(self):
		self.loss_value = None
		self.grads_values = None

	def loss(self, x):
		assert self.loss_value is None
		loss_value, grad_values = eval_loss_and_grads(x)
		self.loss_value = loss_value
		self.grad_values = grad_values
		return self.loss_value

	def grads(self, x):
		assert self.loss_value is not None
		grad_values = np.copy(self.grad_values)
		self.loss_value = None
		self.grad_values = None
		return grad_values

evaluator = Evaluator()

# run scipy-based optimization (L-BFGS) over the pixels of the generated image
# so as to minimize the neural style loss
x = utils.preprocess_image(naive_img_o)

max_iter = 100
for i in range(max_iter):
	print('Start of iteration', i)
	start_time = time.time()
	x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
									 fprime=evaluator.grads, maxfun=20)
	print('Current loss value:', min_val)
	# save current generated image
	img = utils.deprocess_image(x.copy(), img_rows, img_cols)
	save_folder = 'results/' + indices[file_index] + '/'
	os.makedirs(save_folder, exist_ok=True)
	fname = save_folder + 'iteration_%d.png' % i
	save_img(fname, img)
	end_time = time.time()
	print('Image saved as', fname)
	print('Iteration %d completed in %ds' % (i, end_time - start_time))

# ############# ROUGH ######################################

# # define the optimizer and init the setup
# optim_img = Input((fusion_img.shape[0], fusion_img.shape[1], 3))
# # VGG layers to extract features from (first maxpooling layers, see pp. 7 of paper)
# vgg_layers = [3, 6, 10]
# # Get the vgg16 model for perceptual loss
# vgg_weights="imagenet"
# vgg_activations = utils.build_vgg(optim_img, vgg_layers, vgg_weights)
# loss_function = Loss(vgg_activations)

# total_loss = loss_function.loss_total(mask=mask_img)
# # init optim image

# max_iter = 1000
# # first pass - 
# optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
# for itr in range(max_iter):
# 	pdb.set_trace()
