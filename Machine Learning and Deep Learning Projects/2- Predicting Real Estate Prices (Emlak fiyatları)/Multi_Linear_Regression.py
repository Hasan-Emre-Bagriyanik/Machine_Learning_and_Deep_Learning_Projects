import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

# "multilinearregression.csv" dosyasından veri seti okunur ve DataFrame'e yüklenir.
data = pd.read_csv("multilinearregression.csv", sep=";")

# Okunan veri seti ekrana yazdırılır.
print(data)

# Lineer regresyon modeli oluşturulur ve eğitilir.
reg = linear_model.LinearRegression()
reg.fit(data[["alan", "odasayisi", "binayasi"]], data["fiyat"])

# Belirli özelliklere sahip yeni veri noktaları için tahminler yapılır ve sonuçlar ekrana yazdırılır.
print(reg.predict([[230, 4, 10]]))
print(reg.predict([[230, 6, 0]]))
print(reg.predict([[355, 3, 20]]))

# Eğitilmiş modeli "emlakFiyatıTahminEtme.pickle" dosyasına kaydeder.
import pickle
pickle.dump(reg, open("emlakFiyatıTahminEtme.pickle", "wb"))

#%%
# y = a + b1*x1 + b3*x3 + b3*x3
# Modelin katsayıları ve sabiti hesaplanır ve ekrana yazdırılır.
print(reg.coef_)      # Denklemdeki b katsayıları
print(reg.intercept_) # Denklemdeki sabit değer

# Yeni bir veri noktası için el ile tahmin yapılır ve sonuç ekrana yazdırılır.
a = reg.intercept_
b1 = reg.coef_[0]
b2 = reg.coef_[1]
b3 = reg.coef_[2]

x1 = 230
x2 = 4
x3 = 10
y = a + b1 * x1 + b2 * x2 + b3 * x3
print("El ile hesaplanan y değeri:", y)

#%%
# Modelin kaydedilmiş olduğu dosyayı yükleyerek tahminler yapılır ve sonuçlar ekrana yazdırılır.
myModel = pickle.load(open("emlakFiyatıTahminEtme.pickle", "rb"))
print(myModel.predict([[230, 4, 10], [230, 6, 0], [355, 3, 20]]))
