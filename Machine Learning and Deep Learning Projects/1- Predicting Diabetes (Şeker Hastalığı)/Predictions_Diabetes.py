import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Veri setini 'diabetes.csv' dosyasından okuyarak 'data' adlı bir DataFrame'e yükle.
data = pd.read_csv("diabetes.csv")

# Veri setinin ilk beş satırını ekrana yazdır.
data.head()

# Şeker hastaları ve sağlıklı insanlar veri setinden ayrılır.
seker_hastalari = data[data.Outcome == 1]
saglikli_insanlar = data[data.Outcome == 0]

# Yaş ve glukoz değerlerine göre nokta grafiği çizilir ve renklendirilir.
plt.scatter(saglikli_insanlar.Age, saglikli_insanlar.Glucose, color="blue", label="Sağlıklı İnsanlar", alpha=0.4)
plt.scatter(seker_hastalari.Age, seker_hastalari.Glucose, color="red", label="Şeker Hastaları", alpha=0.4)
plt.xlabel("Yaş")
plt.ylabel("Glukoz")
plt.legend()
plt.show()

# Bağımlı değişken 'Outcome' ayrılır ve 'y' değişkenine atanır.
y = data.Outcome.values

# Bağımsız değişkenler 'Outcome' sütunu hariç tutularak 'x_ham_veri' değişkenine atanır.
x_ham_veri = data.drop(["Outcome"], axis=1)

# Min-Max normalizasyonu uygulanır.
x = (x_ham_veri - np.min(x_ham_veri)) / (np.max(x_ham_veri) - np.min(x_ham_veri))

# Veri seti eğitim ve test setlerine ayrılır.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# K-En Yakın Komşular (KNN) sınıflandırıcı modeli oluşturulur ve eğitilir (k=3).
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

# Test seti üzerinde tahmin yapılır ve modelin doğruluğu hesaplanıp yazdırılır.
print("k = 3 için test setinin doğruluk sonucu:", knn.score(x_test, y_test))

# Farklı k değerleri için modelin performansı değerlendirilir.
for k in range(1, 11):
    knn2 = KNeighborsClassifier(n_neighbors=k)
    knn2.fit(x_train, y_train)
    print(f"{k}. k değeri için doğruluk oranı: %{knn2.score(x_test, y_test) * 100}")
    
# en iyi sonuc veren k değeri = 6 

#%%
# Yeni bir hasta tahmini için Min-Max normalizasyonu yapılır ve model üzerinden tahminde bulunulur.
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()
sc.fit_transform(x_ham_veri)

new_prediction = knn.predict(sc.transform(np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])))
print("Yeni bir hastanın tahmini:", new_prediction[0])
