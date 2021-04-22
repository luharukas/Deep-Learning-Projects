#!/usr/bin/env python
# coding: utf-8

# # Chest X-Ray Medical Diagnosis with Deep Learning

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K

from keras.models import load_model

import util


# ##  Load the Datasets

# #### Read in the data
# Let's open these files using the [pandas](https://pandas.pydata.org/) library

# In[ ]:


train_df = pd.read_csv("nih/train-small.csv")
valid_df = pd.read_csv("nih/valid-small.csv")

test_df = pd.read_csv("nih/test.csv")


train_df.head()


# In[ ]:


print(valid_df.shape)


# In[ ]:


labels = ['Cardiomegaly', 
          'Emphysema', 
          'Effusion', 
          'Hernia', 
          'Infiltration', 
          'Mass', 
          'Nodule', 
          'Atelectasis',
          'Pneumothorax',
          'Pleural_Thickening', 
          'Pneumonia', 
          'Fibrosis', 
          'Edema', 
          'Consolidation']


# ### Preventing Data Leakage
# It is worth noting that our dataset contains multiple images for each patient. This could be the case, for example, when a patient has taken multiple X-ray images at different times during their hospital visits. In our data splitting, we have ensured that the split is done on the patient level so that there is no data "leakage" between the train, validation, and test datasets.

# In[ ]:


def check_for_leakage(df1, df2, patient_col):
    
    df1_patients_unique = set(df1[patient_col])
    df2_patients_unique = set(df2[patient_col])
    
    patients_in_both_groups = list(df1_patients_unique.intersection(df2_patients_unique))

    # leakage contains true if there is patient overlap, otherwise false.
    # boolean (true if there is at least 1 patient in both groups)
    if patients_in_both_groups==[]:
        leakage=False
    else:
        leakage=True
    
    return leakage


# In[ ]:


print("leakage between train and test: {}".format(check_for_leakage(train_df, test_df, 'PatientId')))
print("leakage between valid and test: {}".format(check_for_leakage(valid_df, test_df, 'PatientId')))


# <a name='2-2'></a>
# ###  Preparing Images

# In[ ]:


def get_train_generator(df, image_dir, x_col, y_cols, shuffle=True, batch_size=8, seed=1, target_w = 320, target_h = 320):
    print("getting train generator...") 
    # normalize images
    image_generator = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization= True)
    
    # flow from directory with specified batch size
    # and target image size
    generator = image_generator.flow_from_dataframe(
            dataframe=df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            target_size=(target_w,target_h))
    
    return generator


# In[ ]:


def get_test_and_valid_generator(valid_df, test_df, train_df, image_dir, x_col, y_cols, sample_size=100, batch_size=8, seed=1, target_w = 320, target_h = 320):
    print("getting train and valid generators...")
    # get generator to sample dataset
    raw_train_generator = ImageDataGenerator().flow_from_dataframe(
        dataframe=train_df, 
        directory=IMAGE_DIR, 
        x_col="Image", 
        y_col=labels, 
        class_mode="raw", 
        batch_size=sample_size, 
        shuffle=True, 
        target_size=(target_w, target_h))
    
    # get data sample
    batch = raw_train_generator.next()
    data_sample = batch[0]

    # use sample to fit mean and std for test set generator
    image_generator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization= True)
    
    # fit generator to sample from training data
    image_generator.fit(data_sample)

    # get test generator
    valid_generator = image_generator.flow_from_dataframe(
            dataframe=valid_df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h))

    test_generator = image_generator.flow_from_dataframe(
            dataframe=test_df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h))
    return valid_generator, test_generator


# In[ ]:


IMAGE_DIR = "nih/images-small/"
train_generator = get_train_generator(train_df, IMAGE_DIR, "Image", labels)
valid_generator, test_generator= get_test_and_valid_generator(valid_df, test_df, train_df, IMAGE_DIR, "Image", labels)


# In[ ]:


x, y = train_generator.__getitem__(0)
plt.imshow(x[0]);


# <a name='3'></a>
# ##  Model Development
# 
# Now we'll move on to model training and development. We have a few practical challenges to deal with before actually training a neural network, though. The first is class imbalance.

# <a name='3-1'></a>
# ###  Addressing Class Imbalance
# One of the challenges with working with medical diagnostic datasets is the large class imbalance present in such datasets. Let's plot the frequency of each of the labels in our dataset:

# In[ ]:


plt.xticks(rotation=90)
plt.bar(x=labels, height=np.mean(train_generator.labels, axis=0))
plt.title("Frequency of Each Class")
plt.show()


# <a name='Ex-2'></a>
# ###  Computing Class Frequencies

# In[ ]:


def compute_class_freqs(labels):
    
    # total number of patients (rows)
    N = len(labels)
    
    positive_frequencies = np.sum(labels==1,axis=0)/N
    negative_frequencies = np.sum(labels==0,axis=0)/N
    
    return positive_frequencies, negative_frequencies


# In[ ]:


freq_pos, freq_neg = compute_class_freqs(train_generator.labels)
freq_pos
freq_neg


# In[ ]:


data = pd.DataFrame({"Class": labels, "Label": "Positive", "Value": freq_pos})
data = data.append([{"Class": labels[l], "Label": "Negative", "Value": v} for l,v in enumerate(freq_neg)], ignore_index=True)
plt.xticks(rotation=90)
f = sns.barplot(x="Class", y="Value", hue="Label" ,data=data)


# In[ ]:


pos_weights = freq_neg
neg_weights = freq_pos
pos_contribution = freq_pos * pos_weights 
neg_contribution = freq_neg * neg_weights


# In[ ]:


data = pd.DataFrame({"Class": labels, "Label": "Positive", "Value": pos_contribution})
data = data.append([{"Class": labels[l], "Label": "Negative", "Value": v} 
                        for l,v in enumerate(neg_contribution)], ignore_index=True)
plt.xticks(rotation=90)
sns.barplot(x="Class", y="Value", hue="Label" ,data=data);


# In[ ]:



def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
    def weighted_loss(y_true, y_pred):
    
        # initialize loss to zero
        loss = 0.0
        
        for i in range(len(pos_weights)):
            # for each class, add average weighted loss for that class 
            loss +=-1*K.mean((pos_weights[i]*y_true[:,i]*K.log(y_pred[:,i]+epsilon))+(neg_weights[i]*(1-y_true[:,i])*K.log(1-y_pred[:,i]+epsilon)))#complete this line
        return loss

    return weighted_loss


# <a name='3-3'></a>
# ### DenseNet121
# 
# Next, we will use a pre-trained [DenseNet121](https://www.kaggle.com/pytorch/densenet121) model which we can load directly from Keras and then add two layers on top of it:
# 1. A GlobalAveragePooling2D layer to get the average of the last convolution layers from DenseNet121.
# 2. A Dense layer with sigmoid activation to get the prediction logits for each of our classes.
# 
# We can set our custom loss function for the model by specifying the loss parameter in the compile() function.

# In[ ]:


# create the base pre-trained model
base_model = DenseNet121(weights='./nih/densenet.hdf5', include_top=False)

x = base_model.output

# add a global spatial average pooling layer
x = GlobalAveragePooling2D()(x)

# and a logistic layer
predictions = Dense(len(labels), activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss=get_weighted_loss(pos_weights, neg_weights))


# ### Training on the Larger Dataset

# In[ ]:


history = model.fit_generator(train_generator, 
                              validation_data=valid_generator,
                              steps_per_epoch=100, 
                              validation_steps=25, 
                              epochs = 3)

plt.plot(history.history['loss'])
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title("Training Loss Curve")
plt.show()


# In[ ]:


model.load_weights("./nih/pretrained_model.h5")


# <a name='5'></a>
# ##  Prediction and Evaluation

# In[ ]:


predicted_vals = model.predict_generator(test_generator, steps = len(test_generator))


# <a name='5-1'></a>
# ### ROC Curve and AUROC
# 

# In[ ]:


auc_rocs = util.get_roc_curve(labels, predicted_vals, test_generator)


# <a name='5-2'></a>
# ### Visualizing Learning with GradCAM 
# 

# In[ ]:


df = pd.read_csv("nih/train-small.csv")
IMAGE_DIR = "nih/images-small/"

# only show the labels with top 4 AUC
labels_to_show = np.take(labels, np.argsort(auc_rocs)[::-1])[:4]


# In[ ]:


util.compute_gradcam(model, '00008270_015.png', IMAGE_DIR, df, labels, labels_to_show)


# In[ ]:


util.compute_gradcam(model, '00011355_002.png', IMAGE_DIR, df, labels, labels_to_show)


# In[ ]:


util.compute_gradcam(model, '00029855_001.png', IMAGE_DIR, df, labels, labels_to_show)


# In[ ]:


util.compute_gradcam(model, '00005410_000.png', IMAGE_DIR, df, labels, labels_to_show)


# # Task Completed

# # Evaluation of Diagnostic Models

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd  

import util


# In[ ]:


train_results = pd.read_csv("train_preds.csv")
valid_results = pd.read_csv("valid_preds.csv")

# the labels in our dataset
class_labels = ['Cardiomegaly',
 'Emphysema',
 'Effusion',
 'Hernia',
 'Infiltration',
 'Mass',
 'Nodule',
 'Atelectasis',
 'Pneumothorax',
 'Pleural_Thickening',
 'Pneumonia',
 'Fibrosis',
 'Edema',
 'Consolidation']

# the labels for prediction values in our dataset
pred_labels = [l + "_pred" for l in class_labels]


# In[ ]:


y = valid_results[class_labels].values
pred = valid_results[pred_labels].values


# In[ ]:


valid_results[np.concatenate([class_labels, pred_labels])].head()


# In[ ]:


plt.xticks(rotation=90)
plt.bar(x = class_labels, height= y.sum(axis=0));


# In[ ]:


def true_positives(y, pred, th=0.5):
    TP = 0
    # get thresholded predictions
    thresholded_preds = pred >= th
    TP = np.sum((y == 1) & (thresholded_preds == 1))
    
    return TP

def true_negatives(y, pred, th=0.5):
    TN = 0
    thresholded_preds = pred >= th
    # compute TN
    TN = np.sum((y==0) & (thresholded_preds==0))
    return TN

def false_positives(y, pred, th=0.5):
    FP = 0
    
    # get thresholded predictions
    thresholded_preds = pred >= th
    # compute FP
    FP = np.sum((y==0) & (thresholded_preds==1))
    return FP

def false_negatives(y, pred, th=0.5):
    FN = 0
    thresholded_preds = pred >= th
    FN = np.sum((y==1) & (thresholded_preds==0))
    return FN


# In[ ]:


util.get_performance_metrics(y, pred, class_labels)


# In[ ]:


def get_accuracy(y, pred, th=0.5):
    accuracy = 0.0
    TP = true_positives(y, pred, th=0.5)
    FP = false_positives(y,pred,th=0.5)
    TN = true_negatives(y, pred, th=0.5)
    FN = false_negatives(y,pred,th=0.5)

    accuracy = (TP+TN)/(TP+TN+FP+FN)
    return accuracy


# In[ ]:


util.get_performance_metrics(y, pred, class_labels, acc=get_accuracy)


# In[ ]:


get_accuracy(valid_results["Emphysema"].values, np.zeros(len(valid_results)))


# In[ ]:


def get_prevalence(y):
    prevalence = 0.0
    prevalence = np.sum(y==1)/len(y)
    return prevalence


# In[ ]:


util.get_performance_metrics(y, pred, class_labels, acc=get_accuracy, prevalence=get_prevalence)


# In[ ]:


def get_sensitivity(y, pred, th=0.5):
    sensitivity = 0.0
    TP = true_positives(y,pred,th=0.5)
    FN = false_negatives(y,pred,th=0.5)
    sensitivity = TP/(TP+FN)
    
    return sensitivity

def get_specificity(y, pred, th=0.5):
    specificity = 0.0
    TN = true_negatives(y,pred,th=0.5)
    FP = false_positives(y,pred,th=0.5)
    specificity = TN/(TN+FP)
    return specificity


# In[ ]:


# Test
print("Test case")

y_test = np.array([1, 0, 0, 1, 1])
print(f'test labels: {y_test}\n')

preds_test = np.array([0.8, 0.8, 0.4, 0.6, 0.3])
print(f'test predictions: {preds_test}\n')

threshold = 0.5
print(f"threshold: {threshold}\n")

print(f"computed sensitivity: {get_sensitivity(y_test, preds_test, threshold):.2f}")
print(f"computed specificity: {get_specificity(y_test, preds_test, threshold):.2f}")


# In[ ]:


util.get_performance_metrics(y, pred, class_labels, acc=get_accuracy, prevalence=get_prevalence, 
                        sens=get_sensitivity, spec=get_specificity)


# In[ ]:


def get_ppv(y, pred, th=0.5):
    PPV = 0.0
    TP = true_positives(y,pred,th=0.5)
    FP = false_positives(y,pred,th=0.5)
    PPV = TP/(TP+FP)
    return PPV

def get_npv(y, pred, th=0.5):
    NPV = 0.0
    TN = true_negatives(y,pred,th=0.5)
    FN = false_negatives(y,pred,th=0.5)
    NPV = TN/(TN+FN)
    return NPV


# In[ ]:


util.get_performance_metrics(y, pred, class_labels, acc=get_accuracy, prevalence=get_prevalence, 
                        sens=get_sensitivity, spec=get_specificity, ppv=get_ppv, npv=get_npv)


# In[ ]:


util.get_curve(y, pred, class_labels)


# In[ ]:


from sklearn.metrics import roc_auc_score
util.get_performance_metrics(y, pred, class_labels, acc=get_accuracy, prevalence=get_prevalence, 
                        sens=get_sensitivity, spec=get_specificity, ppv=get_ppv, npv=get_npv, auc=roc_auc_score)


# In[ ]:


def bootstrap_auc(y, pred, classes, bootstraps = 100, fold_size = 1000):
    statistics = np.zeros((len(classes), bootstraps))

    for c in range(len(classes)):
        df = pd.DataFrame(columns=['y', 'pred'])
        df.loc[:, 'y'] = y[:, c]
        df.loc[:, 'pred'] = pred[:, c]
        # get positive examples for stratified sampling
        df_pos = df[df.y == 1]
        df_neg = df[df.y == 0]
        prevalence = len(df_pos) / len(df)
        for i in range(bootstraps):
            # stratified sampling of positive and negative examples
            pos_sample = df_pos.sample(n = int(fold_size * prevalence), replace=True)
            neg_sample = df_neg.sample(n = int(fold_size * (1-prevalence)), replace=True)

            y_sample = np.concatenate([pos_sample.y.values, neg_sample.y.values])
            pred_sample = np.concatenate([pos_sample.pred.values, neg_sample.pred.values])
            score = roc_auc_score(y_sample, pred_sample)
            statistics[c][i] = score
    return statistics

statistics = bootstrap_auc(y, pred, class_labels)


# In[ ]:


util.print_confidence_intervals(class_labels, statistics)


# In[ ]:


util.get_curve(y, pred, class_labels, curve='prc')


# In[ ]:


from sklearn.metrics import f1_score
util.get_performance_metrics(y, pred, class_labels, acc=get_accuracy, prevalence=get_prevalence, 
                        sens=get_sensitivity, spec=get_specificity, ppv=get_ppv, npv=get_npv, auc=roc_auc_score,f1=f1_score)


# In[ ]:


from sklearn.calibration import calibration_curve
def plot_calibration_curve(y, pred):
    plt.figure(figsize=(20, 20))
    for i in range(len(class_labels)):
        plt.subplot(4, 4, i + 1)
        fraction_of_positives, mean_predicted_value = calibration_curve(y[:,i], pred[:,i], n_bins=20)
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.plot(mean_predicted_value, fraction_of_positives, marker='.')
        plt.xlabel("Predicted Value")
        plt.ylabel("Fraction of Positives")
        plt.title(class_labels[i])
    plt.tight_layout()
    plt.show()


# In[ ]:


from sklearn.linear_model import LogisticRegression as LR 

y_train = train_results[class_labels].values
pred_train = train_results[pred_labels].values
pred_calibrated = np.zeros_like(pred)

for i in range(len(class_labels)):
    lr = LR(solver='liblinear', max_iter=10000)
    lr.fit(pred_train[:, i].reshape(-1, 1), y_train[:, i])    
    pred_calibrated[:, i] = lr.predict_proba(pred[:, i].reshape(-1, 1))[:,1]


# In[ ]:




