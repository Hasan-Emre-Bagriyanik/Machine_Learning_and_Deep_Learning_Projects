import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from bs4 import BeautifulSoup
import re
import nltk
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords

# Veri setlerimizi yüklüyoruz..
df = pd.read_csv("NLPlabeledData.tsv", delimiter="\t", quoting=3)

# verilerimize bakalım
df.head()

len(df)
len(df["review"])

# stopwords'ü temizlemek için nltk kütüphanesinden stopwords kelime setini bilgisayarımıza indirmemiz gerekiyor. 
# Bu işlemi nltk ile yapıyoruz
nltk.download("stopwords")

#%%  tek bir satır için veri temizleme yaptık 
#  * * * * Veri Temizleme İşlemleri * * * *

#  Öncelikle BeautifulSoup modülünü kullanarak HTML taglerini review cümlelerinden sileceğiz.
# Bu işlemlerin nasıl yapıldığını açıklamak için önce örnek tek bir review seçip size nasıl yapıldığına bakalım:

sample_review = df.review[0]
sample_review

# HTML tagleri temizlendikten sonra..
sample_review =BeautifulSoup(sample_review).get_text()
sample_review

# noktalama işaretleri ve sayılardan temizliyoruz - regex kullanarak..
sample_review = re.sub("[^a-zA-Z]", ' ', sample_review)
sample_review

# küçük harfe dönüştürüyoruz, makine öğrenim algoritmalarımızın büyük harfle başlayan kelimeleri farklı kelime olarak
# algılamaması için yapıyoruz bunu:
sample_review = sample_review.lower()
sample_review

# stopwords (yani the, is, are gibi kelimeler yapay zeka tarafından kullanılmamasını istiyoruz. Bunlar gramer kelimeri..)
# önce split ile kelimeleri bölüyoruz ve listeye dönüştürüyoruz. amacımız stopwords kelimelerini çıkarmak..
sample_review = sample_review.split()
sample_review

len(sample_review)

# sample_review without stopwords
swords = set(stopwords.words("english"))
sample_review = [ w for w in sample_review if w not in swords]
sample_review

len(sample_review)


#%%
# Temizleme işlemini açıkladıktan sonra şimdi tüm dataframe'imiz içinde bulunan reviewleri döngü içinde topluca temizliyoruz
# bu amaçla önce bir fonksiyon oluşturuyoruz:

def process(review):
    # review without HTML tags
    review = BeautifulSoup(review).get_text()
    # review without punctuation and numbers
    review = re.sub("[^a-zA-Z]", ' ', review)
    # converting into lowercase and splitting to eliminate stopwords
    review = review.lower()
    review = review.split()
    # review without stopwords
    swords = set(stopwords.words("english"))
    review = [w for w in review if w not in swords]
    # splitted paragraph'ları space ile birleştiriyoruz return
    return(" ".join(review))


# training datamızı yukardaki fonksiyon yardımıyla temizliyoruz: 
# her 1000 review sonrası bir satır yazdırarak review işleminin durumunu görüyoruz..
train_x_tum = []
for r in range(len(df["review"])):        
    if (r+1)%1000 == 0:        
        print("No of reviews processed =", r+1)
    train_x_tum.append(process(df["review"][r]))  

#%% 

x = train_x_tum
y = np.array(df["sentiment"])

# train test split 
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1, random_state=42)

#  Bag of Words oluşturuyoruz !
# Verilerimizi temizledik ancak yapay zekanın çalışması için bu metin tabanlı verileri sayılara ve bag of words denilen matrise çevirmek gerekiyor
# İşte bu amaçla sklearn içinde bulunan CountVectorizer aracını kullanıyoruz:

# sklearn içinde bulunan countvectorizer fonksiyonunu kullanarak max 5000 kelimelik bag of words oluşturuyoruz...
vectorizer = CountVectorizer(max_features=10000)

# train verilerimizi feature vektöre matrisine çeviriyoruz
x_train = vectorizer.fit_transform(x_train)
x_train

# Bunu array'e dönüştürüyoruz çünkü fit işlemi için array istiyor..
x_train = x_train.toarray()

x_train.shape
y_train.shape

y_train

#%%
#  Random Forest Modeli oluşturuyoruz ve fit ediyoruz

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(x_train, y_train)
#%%
#  Şimdi sıra test datamızda..
# Test verilerimizi feature vektöre matrisine çeviriyoruz
# Yani aynı işlemleri(bag of wordse dönüştürme) tekrarlıyoruz bu sefer test datamız için:
test_x = vectorizer.transform(x_test)
test_x

test_x = test_x.toarray()
test_x.shape

#  Prediction yapıyoruz..
test_predict = model.predict(test_x)
dogruluk = roc_auc_score(y_test, test_predict)

print("Doğruluk oranı: %.2f%%" % (dogruluk * 100))
