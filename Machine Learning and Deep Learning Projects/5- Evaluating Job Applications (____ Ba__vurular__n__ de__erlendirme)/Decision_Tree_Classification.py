import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree

# CSV dosyasını okuyalım ve ilk beş satırını gösterelim
df = pd.read_csv("DecisionTreesClassificationDataSet.csv")
df.head()

# Yapılan düzeltmeler: Kategorik verileri sayısal değerlere dönüştürme
duzeltme_mapping = {"Y": 1, "N": 0}
df["IseAlindi"] = df["IseAlindi"].map(duzeltme_mapping)
df["StajBizdeYaptimi?"] = df["StajBizdeYaptimi?"].map(duzeltme_mapping)
df["Top10 Universite?"] = df["Top10 Universite?"].map(duzeltme_mapping)
df["SuanCalisiyor?"] = df["SuanCalisiyor?"].map(duzeltme_mapping)

duzeltme_mapping_egitimi = {"BS": 0, "MS": 1, "PhD": 2}
df['Egitim Seviyesi'] = df['Egitim Seviyesi'].map(duzeltme_mapping_egitimi)

df.head()

# Bağımlı ve bağımsız değişkenleri belirleyelim
y = df["IseAlindi"]
x = df.drop(["IseAlindi"], axis=1)

x.head()
y.head()

# Decision Tree sınıflandırıcısını oluşturalım ve eğitelim
clf = tree.DecisionTreeClassifier()
clf.fit(x, y)
# Prediction yapalım şimdi
# 5 yıl deneyimli, hazlihazırda bir yerde çalışan ve 3 eski şirkette çalışmış olan, eğitim seviyesi Lisans
# top-tier-school mezunu değil
print(clf.predict([[5,1,3,0,0,0]]))
# Toplam 2 yıllık iş deneyimi, 7 kez iş değiştirmiş çok iyi bir okul mezunu şuan çalışmıyor
print(clf.predict([[2,0,7,0,1,0]]))
# Toplam 2 yıllık iş deneyimi, 7 kez iş değiştirmiş çok iyi bir okul mezunu değil şuan çalışıyor
print(clf.predict([[2,1,7,0,0,0]]))
# Toplam 20 yıllık iş deneyimi, 5 kez iş değiştirmiş iyi bir okul mezunu şuan çalışmıyor
print(clf.predict([[20,0,5,1,1,1]]))