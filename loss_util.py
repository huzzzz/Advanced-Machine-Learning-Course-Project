import keras.backend as K
from keras.applications import VGG16

# ## Contains the different loss function that we are using

# the gram matrix of an image tensor (feature-wise outer product)
def gram_matrix(x):
	features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
	gram = K.dot(features, K.transpose(features))
	return gram

# the "style loss" is designed to maintain
# the style of the reference image in the generated image.
# It is based on the gram matrices (which capture style) of
# feature maps from the style reference image
# and from the generated image


def style_loss(style, combination, img_rows, img_cols):
	assert K.ndim(style) == 3
	assert K.ndim(combination) == 3
	S = gram_matrix(style)
	C = gram_matrix(combination)
	channels = 3
	size = img_rows * img_cols
	# pdb.set_trace()
	return K.sum(K.square(S - C)) / (4.0 * (channels ** 2) * (int(size) ** 2))

# an auxiliary loss function
# designed to maintain the "content" of the
# base image in the generated image


def content_loss(base, combination):
	return K.sum(K.square(combination - base))

# the 3rd loss function, total variation loss,
# designed to keep the generated image locally coherent


def total_variation_loss(x, img_rows, img_cols):
	assert K.ndim(x) == 4
	a = K.square(
		x[:, :img_rows - 1, :img_cols - 1, :] - x[:, 1:, :img_cols - 1, :])
	b = K.square(
		x[:, :img_rows - 1, :img_cols - 1, :] - x[:, :img_rows - 1, 1:, :])
	return K.sum(K.pow(a + b, 1.25))
