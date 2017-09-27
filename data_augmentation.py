
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
# import tensorflow as tf
from keras.preprocessing import image
from os.path import join
import matplotlib.pyplot as plt

input_size = 1024

# data_dir = 'dataset/'
np.random.seed(1987)
orig_width = 1918
orig_height = 1280

'''
df_train = pd.read_csv(join(data_dir, 'train_masks.csv'), usecols=['img'])
df_train['img_id'] = df_train['img'].map(lambda s: s.split('.')[0])
# df_train.head(3)


def get_image_and_mask(img_id):
    img = image.load_img(join(data_dir, 'train', '%s.jpg' % img_id),
                         target_size=(input_size, input_size))
    img = image.img_to_array(img)
    mask = image.load_img(join(data_dir, 'train_masks', '%s_mask.gif' % img_id),
                          grayscale=True, target_size=(input_size, input_size))
    mask = image.img_to_array(mask)
    img, mask = img / 255., mask / 255.
    return img, mask
'''

def plot_img_and_mask(img, mask):
    fig, axs = plt.subplots(ncols=2, figsize=(10, 5), sharex=True, sharey=True)
    axs[0].imshow(img)
    axs[1].imshow(mask[:, :, 0])
    for ax in axs:
        ax.set_xlim(0, input_size)
        ax.axis('off')
    fig.tight_layout()
    plt.show()

# img_ids = df_train['img_id'].values
# np.random.shuffle(img_ids)
# img_id = img_ids[0]
# img, mask = get_image_and_mask(img_id)
# print img.shape, mask.shape

# print "plotting image"
# plot_img_and_mask(img, mask)

def plot_img_and_mask_transformed(img, mask, img_tr, mask_tr):
    fig, axs = plt.subplots(ncols=4, figsize=(16, 4), sharex=True, sharey=True)
    axs[0].imshow(img)
    axs[1].imshow(mask[:, :, 0])
    axs[2].imshow(img_tr)
    axs[3].imshow(mask_tr[:, :, 0])
    for ax in axs:
        ax.set_xlim(0, input_size)
        ax.axis('off')
    fig.tight_layout()
    plt.show()

def random_flip(img, mask, u=0.5):
    if np.random.random() < u:
        img = image.flip_axis(img, 1)
        mask = image.flip_axis(mask, 1)
    return img, mask

# img_flip, mask_flip = random_flip(img, mask, u=1)
# plot_img_and_mask_transformed(img, mask, img_flip, mask_flip)

def rotate(x, theta, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = image.transform_matrix_offset_center(rotation_matrix, h, w)
    x = image.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x

def random_rotate(img, mask, rotate_limit=(-20, 20), u=0.5):
    if np.random.random() < u:
        theta = np.pi / 180 * np.random.uniform(rotate_limit[0], rotate_limit[1])
        img = rotate(img, theta)
        mask = rotate(mask, theta)
    return img, mask

rotate_limit=(-30, 30)
# theta = np.pi / 180 * np.random.uniform(rotate_limit[0], rotate_limit[1])
# print 'theta %.2f' % theta
# img_rot = rotate(img, theta)
# mask_rot = rotate(mask, theta)
# plot_img_and_mask_transformed(img, mask, img_rot, mask_rot)

def shift(x, wshift, hshift, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    h, w = x.shape[row_axis], x.shape[col_axis]
    tx = hshift * h
    ty = wshift * w
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])
    transform_matrix = translation_matrix  # no need to do offset
    x = image.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x

def random_shift(img, mask, w_limit=(-0.1, 0.1), h_limit=(-0.1, 0.1), u=0.5):
    if np.random.random() < u:
        wshift = np.random.uniform(w_limit[0], w_limit[1])
        hshift = np.random.uniform(h_limit[0], h_limit[1])
        img = shift(img, wshift, hshift)
        mask = shift(mask, wshift, hshift)
    return img, mask

w_limit=(-0.2, 0.2)
h_limit=(-0.2, 0.2)
# wshift = np.random.uniform(w_limit[0], w_limit[1])
# hshift = np.random.uniform(h_limit[0], h_limit[1])
# print 'wshift: %.2f, hshift: %.2f' % (wshift, hshift)
# img_shift = shift(img, wshift, hshift)
# mask_shift = shift(mask, wshift, hshift)
# plot_img_and_mask_transformed(img, mask, img_shift, mask_shift)

def zoom(x, zx, zy, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = image.transform_matrix_offset_center(zoom_matrix, h, w)
    x = image.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x

def random_zoom(img, mask, zoom_range=(0.8, 1), u=0.5):
    if np.random.random() < u:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
        img = zoom(img, zx, zy)
        mask = zoom(mask, zx, zy)
    return img, mask

zoom_range=(0.7, 1)
# zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
# print 'zx: %.2f, zy: %.2f' % (zx, zy)
# img_zoom = zoom(img, zx, zy)
# mask_zoom = zoom(mask, zx, zy)
# plot_img_and_mask_transformed(img, mask, img_zoom, mask_zoom)

def shear(x, shear, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = image.transform_matrix_offset_center(shear_matrix, h, w)
    x = image.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x

def random_shear(img, mask, intensity_range=(-0.5, 0.5), u=0.5):
    if np.random.random() < u:
        sh = np.random.uniform(-intensity_range[0], intensity_range[1])
        img = shear(img, sh)
        mask = shear(mask, sh)
    return img, mask

intensity = 0.5
# sh = np.random.uniform(-intensity, intensity)
# print 'sh: %.2f' % sh
# img_shear = shear(img, sh)
# mask_shear = shear(mask, sh)
# plot_img_and_mask_transformed(img, mask, img_shear, mask_shear)

def plot_img_transformed(img, img_tr):
    fig, axs = plt.subplots(ncols=2, figsize=(10, 5), sharex=True, sharey=True)
    axs[0].imshow(img)
    axs[1].imshow(img_tr)
    for ax in axs:
        ax.set_xlim(0, input_size)
        ax.axis('off')
    fig.tight_layout()
    plt.show()

def random_channel_shift(x, limit, channel_axis=2):
    x = np.rollaxis(x, channel_axis, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [np.clip(x_ch + np.random.uniform(-limit, limit), min_x, max_x) for x_ch in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x

# img_chsh = random_channel_shift(img, limit=0.05)
# plot_img_transformed(img, img_chsh)

def random_gray(img, u=0.5):
    if np.random.random() < u:
        coef = np.array([[[0.114, 0.587, 0.299]]])  # rgb to gray (YCbCr)
        gray = np.sum(img * coef, axis=2)
        img = np.dstack((gray, gray, gray))
    return img

# img_gray = random_gray(img, u=1)
# plot_img_transformed(img, img_gray)

def random_contrast(img, limit=(-0.3, 0.3), u=0.5):
    if np.random.random() < u:
        alpha = 1.0 + np.random.uniform(limit[0], limit[1])
        coef = np.array([[[0.114, 0.587, 0.299]]])  # rgb to gray (YCbCr)
        gray = img * coef
        gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
        img = alpha * img + gray
        img = np.clip(img, 0., 1.)
    return img

# img_contrast = random_contrast(img, u=1)
# plot_img_transformed(img, img_contrast)

def random_brightness(img, limit=(-0.3, 0.3), u=0.5):
    if np.random.random() < u:
        alpha = 1.0 + np.random.uniform(limit[0], limit[1])
        img = alpha * img
        img = np.clip(img, 0., 1.)
    return img

# img_brightness = random_brightness(img, u=1)
# plot_img_transformed(img, img_brightness)

def random_saturation(img, limit=(-0.3, 0.3), u=0.5):
    if np.random.random() < u:
        alpha = 1.0 + np.random.uniform(limit[0], limit[1])
        coef = np.array([[[0.114, 0.587, 0.299]]])
        gray = img * coef
        gray = np.sum(gray, axis=2, keepdims=True)
        img = alpha * img + (1. - alpha) * gray
        img = np.clip(img, 0., 1.)
    return img

# img_sat = random_saturation(img, u=1)
# plot_img_transformed(img, img_sat)

def plot_img_and_mask_transformed3(img, mask, img_tr1, mask_tr1, img_tr2, mask_tr2):
    fig, axs = plt.subplots(ncols=6, figsize=(30, 5), sharex=True, sharey=True)
    axs[0].imshow(img)
    axs[1].imshow(mask[:, :, 0])
    axs[2].imshow(img_tr1)
    axs[3].imshow(mask_tr1[:, :, 0])
    axs[4].imshow(img_tr2)
    axs[5].imshow(mask_tr2[:, :, 0])
    for ax in axs:
        ax.set_xlim(0, input_size)
        ax.axis('off')
    fig.tight_layout()
    plt.show()

def random_augmentation(img, mask):
    img = random_channel_shift(img, limit=0.05)
    img = random_brightness(img, limit=(-0.5, 0.5), u=0.5)
    img = random_contrast(img, limit=(-0.5, 0.5), u=0.5)
    img = random_saturation(img, limit=(-0.5, 0.5), u=0.5)
    img = random_gray(img, u=0.2)
    img, mask = random_rotate(img, mask, rotate_limit=(-20, 20), u=0.5)
    img, mask = random_shear(img, mask, intensity_range=(-0.3, 0.3), u=0.2)
    img, mask = random_flip(img, mask, u=0.3)
    img, mask = random_shift(img, mask, w_limit=(-0.1, 0.1), h_limit=(-0.1, 0.1), u=0.3)
    img, mask = random_zoom(img, mask, zoom_range=(0.8, 1), u=0.3)
    return img, mask

def augment(img, mask):
    # img, mask = get_image_and_mask(img_id)
    img_aug1, mask_aug1 = random_augmentation(img, mask)
    img_aug2, mask_aug2 = random_augmentation(img, mask)
    
if __name__ == '__main__':
    augment(img, mask)

'''
for img_id in img_ids[:16]:
    img, mask = get_image_and_mask(img_id)
    img_aug1, mask_aug1 = random_augmentation(img, mask)
    img_aug2, mask_aug2 = random_augmentation(img, mask)
    # plot_img_and_mask_transformed3(img, mask, img_aug1, mask_aug1, img_aug2, mask_aug2)
'''

