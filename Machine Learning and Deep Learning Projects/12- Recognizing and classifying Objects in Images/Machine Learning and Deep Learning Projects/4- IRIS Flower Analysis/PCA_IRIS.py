import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Veriyi okuyup başlıkları ekleyelim
df = pd.read_csv("pca_iris.data", names=["sepal length", "sepal width", "petal length", "petal width", "target"])

# Veri kümesinin ilk 5 satırını görelim
print(df.head())

# Veri setini bağımsız ve bağımlı değişkenlere ayıralım
x = df[["sepal length", "sepal width", "petal length", "petal width"]]
y = df[["target"]]

# Bağımsız değişkenleri standartlaştıralım
x = StandardScaler().fit_transform(x)
print(x)

# PCA modelini tanımlayalım ve bileşenleri dönüştürelim
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDF = pd.DataFrame(data=principalComponents, columns=["principal component 1", "principal component 2"])

# Yeni bileşenleri içeren DataFrame'i oluşturalım ve ilk 5 satırını görelim
finalDataFrame = pd.concat([principalDF, df[["target"]]], axis=1)
print(finalDataFrame.head())

# Hedef sınıflara göre veriyi ayıralım
dfSetosa = finalDataFrame[df.target == "Iris-setosa"]
dfVirginica = finalDataFrame[df.target == "Iris-virginica"]
dfVersicolor = finalDataFrame[df.target == "Iris-versicolor"]

# Grafikle görselleştirelim
plt.xlabel("principal component 1")
plt.ylabel("principal component 2")

plt.scatter(dfSetosa["principal component 1"], dfSetosa["principal component 2"], color="green")
plt.scatter(dfVirginica["principal component 1"], dfVirginica["principal component 2"], color="red")
plt.scatter(dfVersicolor["principal component 1"], dfVersicolor["principal component 2"], color="blue")

#%%
# Alternatif görselleştirme yöntemi
targets = ["Iris-setosa", "Iris-virginica", "Iris-versicolor"]
colors = ["b", "r", "g"]

plt.xlabel("principal component 1")
plt.ylabel("principal component 2")

for target, color in zip(targets, colors):
    temp_df = finalDataFrame[finalDataFrame["target"] == target]
    plt.scatter(temp_df["principal component 1"], temp_df["principal component 2"], color=color)

# PCA'nın açıkladığı varyans oranlarını hesaplayalım
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())

# Sonuçları gösterelim
plt.show()
