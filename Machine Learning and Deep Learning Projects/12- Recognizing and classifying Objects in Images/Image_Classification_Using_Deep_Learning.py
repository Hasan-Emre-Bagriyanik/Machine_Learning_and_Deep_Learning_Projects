
# Deep Learning Algoritmaları ile Fotoğraflardaki Nesneleri Tanıma ve Sınıflandırma Projesi\n",
    
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D



# Datasetimizi (cifar10 verisetini) yüklüyoruz: (Yükleme işlemi için Internet bağlantınızın olması gerekiyor). Eğer bağlantınız yoksa  veri setini Internetten indirip de yükleyebilirsiniz..

(x_train, y_train),(x_test,y_test) = datasets.cifar10.load_data()

#%%

x_train.shape
# Her bir fotoğraf 32 pixele-32 pixel kare boyutunda ve renkli 3 kanal RGB bilgileri olduğu için arrayımız bu şekilde

x_test.shape

y_train[:3]

""""y_train ve y_test 2 boyutlu bir array olarak tutuluyor cifar10 verisetinde. \n",
    "Biz bu verileri görsel olarak daha rahat anlamak için tek boyutlu hale getiriyoruz.\n",
    "2 boyutlu bir arrayi (sadece tekbir boyutunda veri var diğer boyutu boş olan tabi) tekboyutlu hale geitrmek için reshape() kullanıyoruz.."""

y_test = y_test.reshape(-1,)
y_test
#%%
# Verilere bir göz atalım. bu amaçla kendimiz bir array oluşturuyoruz:
resim_siniflari = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

def plot_sample(x,y,index):
    plt.figure(figsize=(3,2))
    plt.imshow(x[index])
    plt.xlabel(resim_siniflari[y[index]])
    plt.show()

plot_sample(x_test, y_test, 0)
plot_sample(x_test, y_test, 1)
plot_sample(x_test, y_test, 2)
plot_sample(x_test, y_test, 2000)

#%%  Normalization

# Verilerimizi normalize etmemiz gerekiyor. Aksi takdirde CNN algoritmaları yanlış sonuç verebiliyor. 
# Fotoğraflar RGB olarak 3 kanal ve her bir pixel 0-255 arasında değer aldığı için normalization için basitçe her bir pixel değerini 255'e bölmemiz yeterli..

x_train = x_train / 255
x_test = x_test / 255

#%% 
# Deep Learning Algoritmamızı CNN - Convolutional Neural Network Kullanarak Tasarlıyoruz:
     
model = Sequential()
# İlk bölüm Convolution layer.. Bu kısımda fotoğraflardan tanımlama yapabilmek için özellikleri çıkarıyoruz
model.add(Conv2D(filters = 32, kernel_size = (3,3), activation="relu", input_shape = (32,32,3)))
model.add(MaxPool2D(pool_size = (2,2)))

model.add(Conv2D(filters = 64, kernel_size = (3,3), activation="relu"))
model.add(MaxPool2D(pool_size = (2,2)))

# İkinci bölüm klasik Articial Neural Network olan layerımız.. Yukarıdaki özelliklerimiz ve training bilgilerine\
# fully connected
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dense(256, activation="relu"))
model.add(Dense(10, activation  ="softmax"))

model.compile(optimizer = "adam", loss  ="sparse_categorical_crossentropy", metrics=["accuracy"] )

model.fit(x_train, y_train, epochs = 20,batch_size = 64)

#%%
# Modelin performansını değerlendirme
model.evaluate(x_test, y_test)

# Modelin test veri seti üzerinde tahminler yapması
y_pred = model.predict(x_test)
y_pred[:3]  # İlk üç tahmin sonucunu görüntüler

# Tahmin edilen sınıfları elde etmek için en yüksek olasılığa sahip indeksi bulma
y_prediction_siniflari = [np.argmax(element) for element in y_pred]
y_prediction_siniflari[:3]  # İlk üç tahmin edilen sınıfı görüntüler

# İlk üç test örneğini görselleştirme
plot_sample(x_test, y_test, 0)  # İlk örneği görselleştirir
resim_siniflari[y_prediction_siniflari[0]]  # İlk örneğin tahmin edilen sınıfını görüntüler

plot_sample(x_test, y_test, 1)  # İkinci örneği görselleştirir
resim_siniflari[y_prediction_siniflari[1]]  # İkinci örneğin tahmin edilen sınıfını görüntüler

plot_sample(x_test, y_test, 2)  # Üçüncü örneği görselleştirir
resim_siniflari[y_prediction_siniflari[2]]  # Üçüncü örneğin tahmin edilen sınıfını görüntüler

