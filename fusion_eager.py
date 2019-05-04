import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.contrib.layers import avg_pool2d
from tensorflow.python.keras import models, losses, layers 
import keras.backend as K
from keras.callbacks import TensorBoard
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications import vgg19, VGG16
import matplotlib.pyplot as plt
import numpy as np
import pdb
import os
import cv2
import utils
import loss_util
import time
import sys
import copy 
from decimal import Decimal
tf.enable_eager_execution()
np.set_printoptions(precision=2, suppress=True)

# copy data
data_dir = 'data/'
output_dir = sys.argv[2]
file_index = int(sys.argv[1])
lb = 5

content_weight = 20
style_weight = 0.5
tv_weight = 0.8

style_img = cv2.imread(data_dir + 'style/' + str(file_index) + '_target.jpg')
mask_img = cv2.imread(data_dir + 'mask/' + str(file_index) + '_c_mask.jpg')
mask_dilated_img = cv2.imread(data_dir + 'mask_dilated/' + str(file_index) + '_c_mask_dilated.jpg')
naive_img = cv2.imread(data_dir + 'fusion/' + str(file_index) + '_naive.jpg')

img_rows, img_cols = naive_img.shape[0] , naive_img.shape[1]
naive_img = utils.preprocess_img(naive_img)
style_img = utils.preprocess_img(style_img)
mask_img_old = mask_img.copy() / 255.0
mask_img = mask_dilated_img
mask_img = tf.expand_dims(mask_img, axis=0) / 255.0

# particular case - 

content_layers = ['block5_conv2'] 
# Style layer we are interested in
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1'
               ]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


def get_model():
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    # Get output layers corresponding to style and content layers 
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    model_outputs = style_outputs + content_outputs
    # Build model 
    return models.Model(vgg.input, model_outputs)

def get_feature_representations(model, content_img, style_img):

    # batch compute content and style features
    style_outputs = model(style_img)
    content_outputs = model(content_img)

    # Get the style and content feature representations from our model  
    style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
    content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]
    return style_features, content_features

def compute_loss(model, loss_weights, fusion_img, style_features, content_features):

    global mask_img
    style_weight, content_weight, tv_weight = loss_weights
    model_outputs = model(fusion_img)
    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]

    style_score = 0
    ustyle_score = 0
    content_score = 0
    ucontent_score = 0

    content_ncc_score = 0
    style_ncc_score = 0
    # style loss
    weight_per_style_layer = 1.0 / float(num_style_layers)
    for target_style, comb_style in zip(style_features, style_output_features):

        local_mask_img = copy.deepcopy(mask_img)
        local_mask_img = tf.image.resize_nearest_neighbor(local_mask_img,  
                [int(target_style.shape[0]),int(target_style.shape[1])])
        ustyle_score += weight_per_style_layer * \
                    loss_util.style_loss(comb_style[0], target_style, img_rows, img_cols)

        style_score += weight_per_style_layer * \
                    loss_util.masked_style_loss(comb_style[0], target_style,local_mask_img, img_rows, img_cols)

        # style_ncc_score += weight_per_style_layer * \
        #             loss_util.ncc_style_loss(comb_style[0], target_style,local_mask_img, img_rows, img_cols)

    # Accumulate content losses from all layers 
    weight_per_content_layer = 1.0 / float(num_content_layers)
    for target_content, comb_content in zip(content_features, content_output_features):
        # downsample mask to match layer dimension
        local_mask_img = copy.deepcopy(mask_img)
        local_mask_img = tf.image.resize_nearest_neighbor(local_mask_img,  
                [int(target_content.shape[0]),int(target_content.shape[1])])
        ucontent_score += weight_per_content_layer* loss_util.content_loss(comb_content[0], target_content)
        content_score += weight_per_content_layer* loss_util.masked_content_loss(comb_content[0],
             target_content, local_mask_img)
        content_ncc_score += weight_per_content_layer* loss_util.ncc_content_loss(comb_content[0],
             target_content, local_mask_img)
    style_score *= style_weight
    content_score *= content_weight
    ustyle_score *= style_weight
    content_ncc_score *= content_weight

    total_variation_loss = tv_weight * loss_util.total_variation_loss(fusion_img, img_rows, img_cols)

    # Get total loss
    print("style loss = {:.2e}".format(style_score.numpy()))
    print("ustyle loss = {:.2e}".format(ustyle_score.numpy()))
    print("content score = {:.2e}".format(content_score.numpy()))
    print("ucontent score = {:.2e}".format(ucontent_score.numpy()))
    print("total variation loss = {:.2e}".format(total_variation_loss.numpy()))
    print("content ncc score = {:.2e}".format(content_ncc_score.numpy()))
    
    loss = 0.5*ustyle_score + 0.5*style_score + content_score + total_variation_loss
    return loss, 0.5*ustyle_score + 0.5*style_score, content_score
    #loss = 0.9*ustyle_score + 0.1*style_score + content_score + total_variation_loss

    # loss = 1.0*ustyle_score + 0.0*style_score - content_ncc_score + total_variation_loss
    # return loss, 1.0*ustyle_score + 0.0*style_score, -content_ncc_score

    # loss = -style_ncc_score - content_ncc_score + total_variation_loss
    # return loss, -style_ncc_score, -content_ncc_score


def compute_grads(cfg):
    with tf.GradientTape() as tape: 
        all_loss = compute_loss(**cfg)
    # Compute gradients wrt input img
    total_loss = all_loss[0]
    return tape.gradient(total_loss, cfg['fusion_img']), all_loss


model = get_model() 
for layer in model.layers:
    layer.trainable = False

# Get the style and content feature representations (from our specified intermediate layers) 
style_features, content_features = get_feature_representations(model, naive_img, style_img)

# Set initial img
if len(sys.argv)==4:
    print("Continued \n")
    init_file = sys.argv[3]
    initial_naive_img = cv2.imread(init_file)
    initial_naive_img = utils.preprocess_img(initial_naive_img)
    fusion_img = copy.deepcopy(initial_naive_img)
else:
    print("Fresh start \n")
    fusion_img = copy.deepcopy(naive_img)

fusion_img = tfe.Variable(fusion_img, dtype=tf.float32)
# Create our optimizer
opt = tf.train.AdamOptimizer(learning_rate=lb)

# opt = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)

# Create a nice config 
loss_weights = (style_weight, content_weight, tv_weight)
cfg = {
  'model': model,
  'loss_weights': loss_weights,
  'fusion_img': fusion_img,
  'style_features': style_features,
  'content_features': content_features
}
blurred_mask = mask_img.numpy()
blurred_mask = blurred_mask[0,:,:,0]
blurred_mask = cv2.GaussianBlur(blurred_mask, (5,5), 1)
# pdb.set_trace()
blurred_mask = np.expand_dims(np.tile(np.expand_dims(blurred_mask, axis=-1) , 3), axis=0)
# For displaying
initial_iter = 0
max_iter = 10000

best_loss = float('inf')
imgs = []
for i in range(initial_iter,max_iter):
    start_time = time.time()
    grads, all_loss = compute_grads(cfg)
    loss, style_score, content_score = all_loss
    grads = grads * mask_img
    opt.apply_gradients([(grads, fusion_img)])
    end_time = time.time()

    print("iteration num: and loss: {:03d}  {:.2e}".format(i, loss.numpy()))
    print()
    print()
    if i%10 == 0:
        if loss < best_loss:
            best_loss = loss
            img = fusion_img.numpy()
            new_mask = np.expand_dims(mask_img_old, axis=0)
            # img = blurred_mask*img + (1-blurred_mask)*style_img
            img = img*new_mask + style_img*(1 - new_mask)
            img = utils.deprocess_img(img, img_rows, img_cols)
            save_folder = output_dir + str(file_index) + '/'
            os.makedirs(save_folder, exist_ok=True)
            fname = save_folder + 'iteration_%d.png' % i
            save_img(fname, img)
            end_time = time.time()
            print('img saved as', fname)
            print('Iteration %d completed in %ds' % (i, end_time - start_time))        
