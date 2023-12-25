from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# First we will define our train and test image paths
train_files_path = "airplanedataset/Train/"
test_files_path = "airplanedataset/Test/"

# let's load any airplane image from our dataset
img = load_img(test_files_path + "B-52/3-1.jpg")

# What is the size
print(img_to_array(img).shape)

# Let's see this airplane, check if we can see it correctly
plt.imshow(img)
plt.show()

#%%
# Let's build our train and test datasets from the directories of airplane images..
train_data = ImageDataGenerator().flow_from_directory(train_files_path, target_size = (224,224))
test_data = ImageDataGenerator().flow_from_directory(test_files_path, target_size  =(224,224))

numberOfAirplaneTypes = 5   # if you have added other planes types

#%%
# Building the Model 
# We will use Transfer Learning, specially VGG16 model for our project!..
# Bur VGG16 model has its own inputs both for training and test, therefore we should change the inputs.
# Original VGG16 model is designed for ImageNet dataset (which is a dataset of over 15 million labeled
#  high-resolution images belonging to roughly 22,000 categories) which has 1000 image categories which are not specifically aircraft images.
#  This means we will build a new model for classificiation of aircraft images in our dataset and use VGG16 pre-trained layers in this new model.

# Let's build our model objects
vgg = VGG16()
vgg_layers = vgg.layers
print(vgg_layers)

#%% 
"""
    I'm gonna build a new Sequential model and I will add the all the layers from the Vgg16 model to my new model except 
     the last layer which is the output layer! Because I will build my own output layer according to my 
     input classes (which are the types of my military aircrafts)... 
     For this I define vggmodel_layersize_tobe_used = len(vgg_layers) - 1 (minus 1 means I omit the last layer - the output layer)
"""

vggmodel_layersize_tobe_used = len(vgg_layers) -1
model = Sequential()
for i in range(vggmodel_layersize_tobe_used):
    model.add(vgg_layers[i])
    
#%% 
# Since I don't want to re-train all the original 16 layers of VGG16
# which has about 138 million (approx) parameters. VGG model has good train parameters, I will use them!!
for layers in model.layers:
    layers.trainable = False

# Since I have omitted the original output layer of VGG16, I have to add my new output layer to my new model
model.add(Dense(numberOfAirplaneTypes, activation="softmax"))

print(model.summary())

#%%
# After model design is complete, it's time to compile
model.compile(loss = "categorical_crossentropy",
              optimizer = "rmsprop",
              metrics = ["accuracy"])

batch_size  = 4

model.fit(train_data,
                    steps_per_epoch = 200 // batch_size,
                    epochs = 20,
                    validation_data = test_data,
                    validation_steps = 200//batch_size)


#%% 
# Let's oad an aircraft image and rescale it to the resolution of 224X224 which VGG16 requires
img = Image.open("15-1.jpg").resize((224,224))

# we must convert it to array fo operations
img = np.array(img)

# Let's look it's shape
img.shape
print(img.ndim)

# We have to add axtra dimension to our array so we will reshape it
img = img.reshape(-1,224,224,3)

# Let's look it's shape
img.shape
print(img.ndim)
#%%

# I will scale input pixels between -1 and 1 using my model's preprocess_input VGG16 model requires it
img = preprocess_input(img)

# Let's see the aircraft 
img_for_display = load_img("15-1.jpg")
plt.imshow(img_for_display)
plt.show()

#%% 
# Let's make a prediction
preds = model.predict(img)
preds

image_classes = ["A-10 Thunderbolt","Boeing B-52","Boeing E-3 Sentry","F-22 Raptor","KC-10 Extender"]

result = np.argmax(preds[0])
print(image_classes[result])
                 