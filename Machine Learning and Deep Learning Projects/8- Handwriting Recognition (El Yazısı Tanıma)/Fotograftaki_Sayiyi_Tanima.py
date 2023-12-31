import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


#%%  her seferinde çalışıtırıldığı zaman tekrar yüklemeye çalışıyor
mnist = fetch_openml("mnist_784")
#%%
mnist.data.shape

#  Mnist veriseti içindeki rakam fotoğraflarını görmek için bir fonksiyon tanımlayalım:

# Parametre olarak dataframe ve ilgili veri fotoğrafının index numarasını alsın..
def showImage(dframe, index):
    
    some_digit = dframe.to_numpy()[index]
    some_digit_image = some_digit.reshape(28,28)
    
    plt.imshow(some_digit_image ,cmap = "binary")
    plt.axis("off")
    plt.show()

# Örnek kullanım:
showImage(mnist.data, 0)
# 70,000 image dosyası, her bir image için 784 boyut(784 feature) mevcut.
#%%

#  Split Data -> Training Set ve Test Set

train_img, test_img, train_lbl, test_lbl = train_test_split(mnist.data, mnist.target, test_size=1/7.0, random_state=0)
type(train_img)

# Rakam tahminlerimizi check etmek için train_img dataframeini kopyalıyoruz, çünkü az sonra değişecek..
test_img_copy = test_img.copy()

showImage(test_img_copy, 2)


#%%
#  Verilerimizi Scale etmemiz gerekiyor:
# 
# Çünkü PCA scale edilmemiş verilerde hatalı sonuçlar verebiliyor bu nedenle mutlaka scaling işleminden geçiriyoruz. 
# Bu amaçla da StandardScaler kullanıyoruz...

scaler = StandardScaler()

# Scaler'ı sadece training set üzerinde fit yapmamız yeterli..
scaler.fit(train_img)

# Ama transform işlemini hem training sete hem de test sete yapmamız gerekiyor..
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)

#%%
#  PCA işlemini uyguluyoruz.. 
# Variance'ın 95% oranında korunmasını istediğimizi belirtiyoruz..
pca = PCA(.95)

# PCA'i sadece training sete yapmamız yeterli 
pca.fit(train_img)

# Şimdi transform işlemiyle hem train hem de test veri setimizin boyutlarını 784'ten 327'e düşürelim:
train_img = pca.transform(train_img)
test_img = pca.transform(test_img)

#%% 

#  2. Aşama
# Şimdi 2. Makine Öğrenmesi modelimiz olan Logistic Regression modelimizi PCA işleminden geçirilmiş veris etimiz üzerinde uygulayacağız.

# default solver çok yavaş çalıştığı için daha hızlı olan 'lbfgs' solverı seçerek logisticregression nesnemizi oluşturuyoruz.
logisticReg = LogisticRegression(solver="lbfgs", max_iter=10000)

#  LogisticRegression Modelimizi train datamızı kullanarak eğitiyoruz:
logisticReg.fit(train_img, train_lbl)

#%%
#  Modelimiz eğitildi şimdi el yazısı rakamları makine öğrenmesi ile tanıma işlemini gerçekletirelim:

logisticReg.predict(test_img[0].reshape(1,-1))
showImage(test_img_copy, 0)

logisticReg.predict(test_img[1].reshape(1,-1))
showImage(test_img_copy, 1)

showImage(test_img_copy, 9900)
logisticReg.predict(test_img[9900].reshape(1,-1))

showImage(test_img_copy, 9999)
logisticReg.predict(test_img[9999].reshape(1,-1))



#  Modelimizin doğruluk oranı (accuracy) ölçmek 
# Modelimizin doğruluk oranı (accuracy) ölçmek için score metodunu kullanacağız:
logisticReg.score(test_img, test_lbl)