# Gerekli kütüphaneler import edilir.
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# CSV dosyasından veri okunur.
df = pd.read_csv("polynomial.csv", sep=";")

# Derecesi 2 olan bir polinom regresyon oluşturulur.
polynomial_regression = PolynomialFeatures(degree=2)

# Veriler görselleştirilir: deneyim-maaş dağılımı scatter plot ile gösterilir ve kaydedilir.
plt.scatter(df["deneyim"], df["maas"])
plt.xlabel("deneyim")
plt.ylabel("maaş")
plt.savefig("1.png", dpi=300)
plt.show()

# Veri, polinom regresyonu için hazırlanır ve polinom regresyon modeli oluşturulur.
x_polynomial = polynomial_regression.fit_transform(df[["deneyim"]])
reg = LinearRegression()
reg.fit(x_polynomial, df["maas"])

# Polinom regresyonu ile elde edilen tahminler ve gerçek değerler görselleştirilir.
y_head = reg.predict(x_polynomial)
plt.plot(df["deneyim"], y_head, color="blue", label="deneyim")
plt.legend()
plt.scatter(df["deneyim"], df["maas"])
plt.show()

#%%
# Rastgele renklerle polinom regresyonu çizdirilir.
def rastgele_renk():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

for i in range(2, 9):
    polynomial_regression = PolynomialFeatures(degree=i)
    x_polynomial = polynomial_regression.fit_transform(df[["deneyim"]])
    reg = LinearRegression()
    reg.fit(x_polynomial, df["maas"])
    
    y_head = reg.predict(x_polynomial)
    plt.plot(df["deneyim"], y_head, color=rastgele_renk(), label="deneyim " + str(i))
    plt.legend()
    plt.scatter(df["deneyim"], df["maas"])
    plt.show()

#%%

# Yeni bir deneyim değeri için maaş tahmini yapılır.
x_polynomial = polynomial_regression.fit_transform([[4.5]])
reg.predict(x_polynomial)

