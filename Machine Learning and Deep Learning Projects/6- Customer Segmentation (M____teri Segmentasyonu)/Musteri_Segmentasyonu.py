import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Veriyi CSV dosyasından oku ve ilk beş satırını göster
df = pd.read_csv("Avm_Musterileri.csv")
df.head()

# 'Annual Income (k$)' ve 'Spending Score (1-100)' değişkenlerinin scatter plotunu çiz
plt.scatter(df["Annual Income (k$)"], df["Spending Score (1-100)"])
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()

# Sütun isimlerini değiştir
df.rename(columns={"Annual Income (k$)": "income"}, inplace=True)
df.rename(columns={"Spending Score (1-100)": "score"}, inplace=True)

# Veriyi ölçeklendirme
scaler = MinMaxScaler()

scaler.fit(df[["income"]])
df["income"] = scaler.transform(df[["income"]])

scaler.fit(df[["score"]])
df["score"] = scaler.transform(df[["score"]])
df.head()

#%%
# Küme sayısının (K) inertia değerine etkisini görmek için farklı K değerleri ile KMeans algoritması uygula
k_range = range(1, 11)
kmeans_list = []

for k in k_range:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df[["income", "score"]])
    kmeans_list.append(kmeans.inertia_)

# Elde edilen inertia değerlerini grafiğe dök
plt.plot(k_range, kmeans_list)
plt.xlabel("K")
plt.ylabel("Distortion değeri (inertia)")
plt.show()

#%%
# En iyi K değeri olarak 5'i seçerek KMeans algoritmasını tekrar uygula ve tahminleri yap
kmeans = KMeans(n_clusters=5)
y_predicted = kmeans.fit_predict(df[["income", "score"]])
y_predicted

# Veri çerçevesine yeni bir 'cluster' sütunu ekle ve tahmin edilen küme numaralarını ata
df["cluster"] = y_predicted
df.head()

# Küme merkezlerini göster
kmeans.cluster_centers_

#%%
# Farklı kümeleri farklı renklerle göster
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]
df4 = df[df.cluster == 3]
df5 = df[df.cluster == 4]

plt.xlabel("income")
plt.ylabel("score")

plt.scatter(df1.income, df1.score, color="blue")
plt.scatter(df2.income, df2.score, color="red")
plt.scatter(df3.income, df3.score, color="green")
plt.scatter(df4.income, df4.score, color="orange")
plt.scatter(df5.income, df5.score, color="black")

# Küme merkezlerini 'X' işaretiyle göster
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color="black", marker="X", label="clustering")
plt.legend()
plt.show()
