# # Image Classification using Transfer Learning

# In this project we will make image classification using Transfer Learning. For this purpose we will use the InceptionResNetV2 model which is trained on the imageNet data set. 
# Lets use a submarine image for classification. I found this image in Internet, this image is not included in imageNet dataset which was used for training InceptionResNetV2.
# 
# In transfer learning we use a model that has been previously trained on a dataset and contains the weights and biases that represent the features of whichever dataset it was trained on. 
# 
# Inception and ResNet have been among the best image recognition performance models in recent years, with very good performance at a relatively low computational cost.
# Inception and ResNet combines the Inception architecture, with residual connections.

import numpy as np
from PIL import Image
from IPython.display import Image as show_image
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions



#  The InceptionResNetV2 pre-trained model expects inputs of 299x299 resolution.  
#  InceptionResNetV2 model will classify images into one of 1,000 possible categories.



# Let's load our image and rescale it to the resolution of 299x299 which InceptionResNetV2 requires..
img = Image.open("sportscar.jpg").resize((299,299))


# We must convert it to array for operations...
img = np.array(img)

img.shape
print(img.ndim)

# We have to add an extra dimension to our array so we will reshape it.. 
img = img.reshape(-1,299,299,3)

img.shape
print(img.ndim)


# I will scale input pixels between -1 and 1 using my model's preprocess_input
# InceptionResNetV2 model requires it..
img = preprocess_input(img)


# Let's load up the model itself:
incresv2_model = InceptionResNetV2(weights="imagenet", classes=1000)   # InceptionResNetV2 will classify images into
# 1,000 possible categories.


#  Lets inspect InceptionResNetV2 model
print(incresv2_model.summary())
print(type(incresv2_model) )


# Before prediction let's see our image with our eyes first:
show_image(filename='sportscar.jpg') 


# It's already trained with weights learned from the Imagenet data set. Now we will use it by calling incresv2_model's predict() method:
preds = incresv2_model.predict(img)
print("Predicted categories: ", decode_predictions(preds,top = 2)[0])
