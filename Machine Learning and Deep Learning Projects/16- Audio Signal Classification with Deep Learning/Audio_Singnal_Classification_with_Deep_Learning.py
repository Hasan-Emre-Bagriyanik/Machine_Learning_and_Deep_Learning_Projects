
# SOUND SIGNAL CLASSIFICATION USING DEEP LEARNING
# This project consists of 3 main steps:
# 1. Step We will prepare our dataset for analysis and extract sound signal features from  audio files using Mel-Frequency Cepstral Coefficients(MFCC).
# 2. Then we will build a Convolutional Neural Networks (CNN) model and train our model with our dataset.
# 3. Finally we predict an audio file's class using our model.

# We will use UrbanSound8K Dataset, download Link is here: https://urbansounddataset.weebly.com/download-urbansound8k.html
# Dataset folder and this source code should be on same directory..
# Don't forget to install librosa library using anaconda promt with the following command line:
# conda install -c conda-forge librosa

#%%
# Step 1: We will prepare our dataset for analysis and extract sound signal features from  audio files using Mel-Frequency Cepstral Coefficients(MFCC)
# Every signal has its own characteristics. In sound processing, the mel-frequency cepstrum (MFC) is a representation of the short-term power spectrum of a sound. 
# Mel-frequency cepstral coefficients (MFCCs) are coefficients that collectively make up an MFC.

import tensorflow as tf
print(tf.__version__)

import matplotlib.pyplot as plt
import pandas as pd
import os
import librosa
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime

#%% 
# First I want to show to how librosa handles sound signals
# Let's read an example audio signal using librosa
audio_file_path = "17973-2-0-32.wav"

librosa_auido_data, librosa_sample_rate = librosa.load(audio_file_path)
print(librosa_auido_data)

# Lets plot the librosa audio data
# Original audio with 1 channel
plt.figure(figsize=(12,4))
plt.plot(librosa_auido_data)
plt.show()

#%%
# Lets read with scipy
from scipy.io import wavfile as wav
wave_sample_rate, wave_audio = wav.read(audio_file_path)
wave_audio

plt.figure(figsize = (12,4))
plt.plot(wave_audio)
plt.show()

#%%
# Feature Extraction
# Here we will be using Mel-Frequency Cepstral Coefficients(MFCC) from the audio samples. The MFCC summarises the frequency distribution across the window size, so it is possible to analyse both the frequency and time characteristics of the sound. 
# These audio representations will allow us to identify features for classification.

mfccs  =librosa.feature.mfcc(y=librosa_auido_data, sr=librosa_sample_rate, n_mfcc=40)
print(mfccs.shape)
mfccs

#%%
# We will extract MFCC's for every audio file in the dataset
audio_dataset_path = "UrbanSound8K/audio"
metadata = pd.read_csv("UrbanSound8K/metadata/UrbanSound8K.csv")
metadata.head()

#%% 
def features_extractor(file_name):
    audio, sample_rate = librosa.load(file_name, res_type="kaiser_fast")
    mfccs_features = librosa.feature.mfcc(y = audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis = 0)
    return mfccs_scaled_features

# Now we iterate through every audio file and extract features 
# using Mel-Frequency Cepstral Coefficients

extracted_features = []
for index_num, row in tqdm(metadata.iterrows()):
    file_name = os.path.join(os.path.abspath(audio_dataset_path),"fold"+str(row["fold"])+ "/" , str(row["slice_file_name"]))
    final_class_labels = row["class"]
    data = features_extractor(file_name)
    extracted_features.append([data, final_class_labels])
    
#%% 
# We will convert extracted_features to Pandas dataframe
extracted_features_df = pd.DataFrame(extracted_features, columns=["feature","class"])
extracted_features_df.head()

#%%
# We then split the dataset into independent and dependent dataset
x = np.array(extracted_features_df["feature"].tolist())
y = np.array(extracted_features_df["class"].tolist())

x.shape

#%%
# We should perform Label Encoding since we need one hot encoded values for output classes in our model (1s and 0s)
labelencoder = LabelEncoder()
y = to_categorical(labelencoder.fit_transform(y))
y

#%%
# We split dataset as Train and Test
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=0, test_size=0.2)

x_train.shape
x_test.shape
y_train.shape
y_test.shape


#%% 
# Step 2: Building a Convolutional Neural Networks (CNN) Model and Train Our Model with UrbanSound8K Dataset.

# How many classes we have? We should  use it in ourm model
num_labels = 10

# Now we start building our CNN model..
model = Sequential()

model.add(Dense(125, input_shape = (40,)))
model.add(Activation("relu"))
model.add(Dropout(0.3))

model.add(Dense(250))
model.add(Activation("relu"))
model.add(Dropout(0.3))

model.add(Dense(125))
model.add(Activation("relu"))

# output layer
model.add(Dense(num_labels))
model.add(Activation("softmax"))

#%%
model.summary()
model.compile(loss = "categorical_crossentropy", metrics = ["accuracy"], optimizer = "adam")

#%%
# training the model
epochs = 150
batch_size = 32

model.fit(x_train, y_train, batch_size =batch_size, epochs = epochs, validation_data = (x_test,y_test), verbose = 1)

#%% 
validation_test_set_accuracy = model.evaluate(x_test, y_test, verbose = 0)
print(validation_test_set_accuracy)

# Tahminlerinizi yapın
predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)

# Doğruluk değerini hesaplayın
accuracy = np.mean(predicted_labels == np.argmax(y_test, axis=1))
print("Test set doğruluğu:", accuracy)

#%% 
# Step 3: Finally We Predict an Audio File's Class Using Our CNN Model
filename = "PoliceSiren.wav"
sound_signal, sample_rate = librosa.load(filename, res_type="kaiser_fast")
mfccs_features = librosa.feature.mfcc(y = sound_signal, sr=sample_rate, n_mfcc=40)
mfccs_slaced_features  =np.mean(mfccs_features.T, axis = 0)

print(mfccs_slaced_features)

mfccs_slaced_features = mfccs_slaced_features.reshape(1,-1)
mfccs_slaced_features.shape

print(mfccs_slaced_features)
print(mfccs_slaced_features.shape)

result_array = model.predict(mfccs_slaced_features)
result_array

result_classes = ["air_conditioner","car_horn","children_playing","dog_bark","drilling", "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]
result = np.argmax(result_array[0])
print(result_classes[result])

