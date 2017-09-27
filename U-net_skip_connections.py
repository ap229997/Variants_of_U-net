
# coding: utf-8

# In[1]:

#get_ipython().magic(u'matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential,Model
from keras.layers import Dense, Conv2D, Input, MaxPool2D, UpSampling2D, Concatenate, Conv2DTranspose, Add
from keras.optimizers import Adam, RMSprop

import tensorflow as tf 

from scipy.misc import imresize
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
from PIL import Image

from keras.preprocessing.image import array_to_img , img_to_array , load_img ,ImageDataGenerator 

from subprocess import check_output
# print check_output(["ls", "../myproject"]).decode("utf8")


# In[2]:

data_dir = "dataset/train/"
mask_dir = "dataset/train_masks/"
all_images = os.listdir(data_dir)


# In[3]:

train_images, validation_images = train_test_split(all_images, train_size=0.8, test_size=0.2)
# train_images[0]
# content_image=Image.open('dataset/train/fc5f1a3a66cf_06.jpg')
# content_image.size


# In[8]:

batch_size = 1
img_size = 512
spe_train = len(train_images)/batch_size
spe_validation = len(validation_images)/batch_size


# In[7]:

def grey2rgb_2(img):
    new_img=np.array(list(img)*3)
    new_img=new_img.reshape(img.shape[0],img.shape[1],3)
    return new_img


# In[13]:

def grey2rgb(img):
    new_img = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_img.append(list(img[i][j])*3)
    new_img = np.array(new_img).reshape(img.shape[0], img.shape[1], 3)
    return new_img


# generator that we will use to read the data from the directory
def data_gen_small(data_dir, mask_dir, images, batch_size, dims):
        """
        data_dir: where the actual images are kept
        mask_dir: where the actual masks are kept
        images: the filenames of the images we want to generate batches from
        batch_size: self explanatory
        dims: the dimensions in which we want to rescale our images
        """
        while True:
            batch = np.random.choice(np.arange(len(images)), batch_size)
            imgs = []
            labels = []
            for i in batch:
                # images
                original_img = load_img(data_dir + images[i])
                resized_img = imresize(original_img, dims+[3])
                array_img = img_to_array(resized_img)/255
                imgs.append(array_img)
                
                # masks
                original_mask = load_img(mask_dir + images[i].split(".")[0] + '_mask.gif')
                resized_mask = imresize(original_mask, dims+[3])
                array_mask = img_to_array(resized_mask)/255
                labels.append(array_mask[:, :, 0])
            imgs = np.array(imgs)
            labels = np.array(labels)
            #print labels
            yield imgs, labels.reshape(-1, dims[0], dims[1], 1)

# example use
train_gen = data_gen_small(data_dir, mask_dir, train_images, batch_size, [img_size, img_size])
img, msk = next(train_gen)

# plt.imshow(img[0])
# plt.imshow(grey2rgb(msk[0]), alpha=0.5)


# In[14]:

# from keras.layers import AvgPool2D

def down(input_layers,filters,pool=True):
    conv1=Conv2D(filters,(2,2),padding="same",activation='relu')(input_layers)
    residual = Conv2D(filters, (3, 3), padding='same', activation='relu')(conv1)
    extra=Conv2D(filters,(1,1),padding="same",activation='relu')(input_layers)
    if pool:
        ext = Add()([residual,extra])
        max_pool = MaxPool2D()(ext)
        return max_pool, residual
    else:
        return residual

def up(input_layer, residual, filters):
    filters=int(filters)
    # upsample = UpSampling2D()(input_layer)
    upsample =Conv2DTranspose(filters,(4,4),padding='same',activation='relu',strides=2)(input_layer)
    upconv = Conv2D(filters, kernel_size=(2, 2), padding="same")(upsample)
    concat = Concatenate(axis=3)([residual, upconv])
    conv1 = Conv2D(filters, (3, 3), padding='same', activation='relu')(concat)
    conv2 = Conv2D(filters, (3, 3), padding='same', activation='relu')(conv1)
    extra=Conv2D(filters,(1,1),padding="same",activation='relu')(upsample)
    ext = Add()([conv2,extra])
    return ext


# In[15]:

filters = 16
input_layer = Input(shape = [512, 512, 3])
layers = [input_layer]
residuals = []


# Adding few extra layers
d00, res00 = down(input_layer, filters)
residuals.append(res00)

filters *= 2

# next
d0, res0 = down(d00, filters)
# d2, res2_cur = down(d1, filters)
# res2_pre = Conv2D(filters,(2,2),padding="same",activation='relu')(res1)
res0 = Concatenate(axis=3)([d00, res0])
residuals.append(res0)

filters *=2


# Down 1, 128
d1, res1 = down(d0, filters)
# res1 = Concatenate(axis=3)([d0, res1])
residuals.append(res1)

filters *= 2

# Down 2, 64
d2, res2 = down(d1, filters)
# d2, res2_cur = down(d1, filters)
# res2_pre = Conv2D(filters,(2,2),padding="same",activation='relu')(res1)
res2 = Concatenate(axis=3)([d1, res2])
residuals.append(res2)

filters *= 2

# Down 3, 32
d3, res3 = down(d2, filters)
# d2, res2_cur = down(d1, filters)
# res2_pre = Conv2D(filters,(2,2),padding="same",activation='relu')(res1)
res3 = Concatenate(axis=3)([d2, res3])
residuals.append(res3)

filters *= 2

# Down 4, 16
d4, res4 = down(d3, filters)
# d2, res2_cur = down(d1, filters)
# res2_pre = Conv2D(filters,(2,2),padding="same",activation='relu')(res1)
res4 = Concatenate(axis=3)([d3, res4])
residuals.append(res4)

filters *= 2

# Down 5, 8
d5 = down(d4, filters, pool=False)

# Up 1, 16
up1 = up(d5, residual=residuals[-1], filters=filters/2)

filters /= 2

# Up 2,  32
up2 = up(up1, residual=residuals[-2], filters=filters/2)

filters /= 2

# Up 3, 64
up3 = up(up2, residual=residuals[-3], filters=filters/2)

filters /= 2

# Up 4, 128
up4 = up(up3, residual=residuals[-4], filters=filters/2)


filters /= 2

# Up 5, 256
up5 = up(up4, residual=residuals[-5], filters=filters/2)

filters /= 2

# Up 6, 512
up6 = up(up5, residual=residuals[-6], filters=filters/2)

out = Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid")(up6)

model = Model(input_layer, out)

#model.summary()


# In[16]:

def dice_coef(y_true, y_pred):
    smooth = 1e-5
    
    y_true = tf.round(tf.reshape(y_true, [-1]))
    y_pred = tf.round(tf.reshape(y_pred, [-1]))
    
    isct = tf.reduce_sum(y_true * y_pred)
    
    return 2 * isct / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred))


# In[ ]:




# In[ ]:

model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=[dice_coef])
# model.fit_generator(train_gen, verbose=1, steps_per_epoch=100, epochs=10)
validation_gen = data_gen_small(data_dir, mask_dir, validation_images, batch_size, [img_size, img_size])
model.fit_generator(train_gen, verbose=1, steps_per_epoch=spe_train, epochs=50, validation_data=validation_gen, validation_steps=spe_validation)


# In[ ]:

model.save_weights('u-net_resnet_style.hdf5')

