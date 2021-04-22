#!/usr/bin/env python
# coding: utf-8

# # Brain Tumor Auto-Segmentation for Magnetic Resonance Imaging (MRI)

# ## Import Packages
# 

# In[ ]:


import keras
import json
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

from tensorflow.keras import backend as K 

import util


# In[ ]:


HOME_DIR = "<file address>"
DATA_DIR = HOME_DIR

def load_case(image_nifty_file, label_nifty_file):
    image = np.array(nib.load(image_nifty_file).get_fdata())
    label = np.array(nib.load(label_nifty_file).get_fdata())
    
    return image, label


# In[ ]:


image, label = load_case(DATA_DIR + "<image name>.nii.gz", DATA_DIR + "<image name>.nii.gz")
image = util.get_labeled_image(image, label)

util.plot_image_grid(image)


# In[ ]:


image, label = load_case(DATA_DIR + "<file name>.nii.gz", DATA_DIR + "<file name>.nii.gz")
util.visualize_data_gif(util.get_labeled_image(image, label))


# ##  Data Preprocessing using patches

# In[ ]:


def get_sub_volume(image, label, 
                   orig_x = 240, orig_y = 240, orig_z = 155, 
                   output_x = 160, output_y = 160, output_z = 16,
                   num_classes = 4, max_tries = 1000, 
                   background_threshold=0.95):
    X = None
    y = None
    tries = 0
    
    while tries < max_tries:
        start_x = np.random.randint(orig_x - output_x + 1)
        start_y = np.random.randint(orig_y - output_y + 1)
        start_z = np.random.randint(orig_z - output_z + 1)
        y = label[start_x: start_x + output_x,
                  start_y: start_y + output_y,
                  start_z: start_z + output_z]
        y = keras.utils.to_categorical(y, num_classes=num_classes)
        bgrd_ratio = y[:,:,:,0].sum()/(output_x*output_y*output_z)
        tries += 1
        if bgrd_ratio < background_threshold:
            X = np.copy(image[start_x: start_x + output_x,
                              start_y: start_y + output_y,
                              start_z: start_z + output_z, :])
            X= np.moveaxis(X,-1,0)
            y =  np.moveaxis(y,-1,0)

            
            y = y[1:, :, :, :]
    
            return X, y
    print(f"Tried {tries} times to find a sub-volume. Giving up...")


# In[ ]:


image, label = load_case(DATA_DIR + "<filename>.nii.gz", DATA_DIR + "<filename>.nii.gz")
X, y = get_sub_volume(image, label)
util.visualize_patch(X[1, :, :, :], y[2])


# In[ ]:


def standardize(image):
    standardized_image = np.zeros(image.shape)
    for c in range(image.shape[0]):
        for z in range(image.shape[3]):
            image_slice = image[c,:,:,z]

            centered = image_slice-np.mean(image_slice)
            
            if np.std(centered) != 0:
                centered_scaled = centered/np.std(centered)
                standardized_image[c, :, :, z] = centered_scaled
    return standardized_image


# In[ ]:


X_norm = standardize(X)
print("standard deviation for a slice should be 1.0")
print(f"stddv for X_norm[0, :, :, 0]: {X_norm[0,:,:,0].std():.2f}")


# In[ ]:


util.visualize_patch(X_norm[0, :, :, :], y[2])


# In[ ]:


def single_class_dice_coefficient(y_true, y_pred, axis=(0, 1, 2), 
    dice_numerator = 2*K.sum(y_true*y_pred)+epsilon
    dice_denominator = K.sum(y_true*y_true)+K.sum(y_pred*y_pred)+epsilon
    dice_coefficient = K.sum(dice_numerator/dice_denominator)
    return dice_coefficient


# In[ ]:



def dice_coefficient(y_true, y_pred, axis=(1, 2, 3), 
                     epsilon=0.00001):
    dice_numerator = 2*K.sum((y_true*y_pred),axis=axis)+epsilon
    dice_denominator = K.sum((y_true*y_true),axis=axis)+K.sum((y_pred*y_pred),axis=axis)+epsilon
    dice_coefficient = K.mean(dice_numerator/dice_denominator)
    return dice_coefficient


# In[ ]:


def soft_dice_loss(y_true, y_pred, axis=(1, 2, 3), 
                   epsilon=0.00001):
    dice_numerator = 2*K.sum((y_true*y_pred),axis=axis)+epsilon
    dice_denominator = K.sum((y_true*y_true),axis=axis)+K.sum((y_pred*y_pred),axis=axis)+epsilon
    dice_loss = 1-(K.mean(dice_numerator/dice_denominator))
    return dice_loss


# In[ ]:


model = util.unet_model_3d(loss_function=soft_dice_loss, metrics=[dice_coefficient])


# In[ ]:


base_dir = HOME_DIR + "processed/"
with open(base_dir + "config.json") as json_file:
    config = json.load(json_file)
train_generator = util.VolumeDataGenerator(config["train"], base_dir + "train/", batch_size=3, dim=(160, 160, 16), verbose=0)
valid_generator = util.VolumeDataGenerator(config["valid"], base_dir + "valid/", batch_size=3, dim=(160, 160, 16), verbose=0)


# In[ ]:


model.load_weights(HOME_DIR + "model_pretrained.hdf5")


# In[ ]:


model.summary()


# #  Evaluation of Overall Performance

# In[ ]:


util.visualize_patch(X_norm[0, :, :, :], y[2])


# In[ ]:


X_norm_with_batch_dimension = np.expand_dims(X_norm, axis=0)
patch_pred = model.predict(X_norm_with_batch_dimension)


# In[ ]:


# set threshold.
threshold = 0.5

# use threshold to get hard predictions
patch_pred[patch_pred > threshold] = 1.0
patch_pred[patch_pred <= threshold] = 0.0


# In[ ]:


print("Patch and ground truth")
util.visualize_patch(X_norm[0, :, :, :], y[2])
plt.show()
print("Patch and prediction")
util.visualize_patch(X_norm[0, :, :, :], patch_pred[0, 2, :, :, :])
plt.show()


# In[ ]:



def compute_class_sens_spec(pred, label, class_num):
    class_pred = pred[class_num]
    class_label = label[class_num]

    tp = np.sum((class_pred==1) & (class_label==1))

    # true negatives
    tn = np.sum((class_pred==0) & (class_label==0))

    
    #false positives
    fp = np.sum((class_pred==1) & (class_label==0))

    
    # false negatives
    fn = np.sum((class_pred==0) & (class_label==1))


    # compute sensitivity and specificity
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    return sensitivity, specificity


# In[ ]:


sensitivity, specificity = compute_class_sens_spec(patch_pred[0], y, 2)

print(f"Sensitivity: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")


# In[ ]:


def get_sens_spec_df(pred, label):
    patch_metrics = pd.DataFrame(
        columns = ['Edema', 
                   'Non-Enhancing Tumor', 
                   'Enhancing Tumor'], 
        index = ['Sensitivity',
                 'Specificity'])
    
    for i, class_name in enumerate(patch_metrics.columns):
        sens, spec = compute_class_sens_spec(pred, label, i)
        patch_metrics.loc['Sensitivity', class_name] = round(sens,4)
        patch_metrics.loc['Specificity', class_name] = round(spec,4)

    return patch_metrics


# In[ ]:


df = get_sens_spec_df(patch_pred[0], y)

print(df)


# In[ ]:


image, label = load_case(DATA_DIR + "imagesTr/BRATS_003.nii.gz", DATA_DIR + "labelsTr/BRATS_003.nii.gz")
pred = util.predict_and_viz(image, label, model, .5, loc=(130, 130, 77))                


# In[ ]:


whole_scan_label = keras.utils.to_categorical(label, num_classes = 4)
whole_scan_pred = pred
whole_scan_label = np.moveaxis(whole_scan_label, 3 ,0)[1:4]
whole_scan_pred = np.moveaxis(whole_scan_pred, 3, 0)[1:4]


# In[ ]:


whole_scan_df = get_sens_spec_df(whole_scan_pred, whole_scan_label)

print(whole_scan_df)


# In[ ]:




