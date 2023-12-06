# NumPy ve Pandas kütüphanelerini içe aktaralım
import numpy as np
import pandas as pd

# Veri çerçevesinin sütun isimlerini tanımlayalım
columns_name = ["user_id", "item_id", "rating", "timestamp"]
# "users.data" adlı dosyayı tab ('\t') ayracı ile okuyup veri çerçevesine yükleyelim
df = pd.read_csv("users.data", sep="\t", names=columns_name)
# Veri çerçevesinin ilk beş satırını görüntüleyelim ve satır sayısını yazdıralım
df.head()
print(len(df))


# "movie_id_titles.csv" adlı dosyayı okuyup veri çerçevesine yükleyelim
movies_titles = pd.read_csv("movie_id_titles.csv")
# Film başlıklarının ilk beş satırını görüntüleyelim ve satır sayısını yazdıralım
movies_titles.head()
print(len(movies_titles))
# "item_id" sütunu üzerinden iki veri çerçevesini birleştirelim ve yeni bir veri çerçevesi oluşturalım
df = pd.merge(df, movies_titles, on="item_id")
# Yeni birleştirilmiş veri çerçevesinin ilk beş satırını gösterelim
df.head()

# Öncelikle Excel'deki pivot tablo benzeri bir yapı kuruyoruz.
# Bu yapıya göre her satır bir kullanıcı olacak şekilde (yani dataframe'imizin index'i user_id olacak)
# Sütunlarda film isimleri (yani title sütunu) olacak,
# tablo içinde de rating değerleri olacak şekilde bir dataframe oluşturuyoruz!
moviemat = pd.pivot_table(data=df, index="user_id", columns="title", values="rating")
moviemat.head()

# Star Wars (1977) filminin user ratinglerine bakalım:
starwars_user_rating = moviemat["Star Wars (1977)"]
starwars_user_rating.head()

# corrwith() metodunu kullanarak Star wars filmi ile korelasyonları hesaplatalım:
smilar_to_starwars = moviemat.corrwith(starwars_user_rating)
smilar_to_starwars
type(smilar_to_starwars)

# Bazı kayıtlarda boşluklar olduğu için hata veriyor similar_to_starwars bir seri, biz bunu corr_starwars isimli bir dataframe'e dönüştürelim 
# ve NaN kayıtlarını temizleyip bakalım:
    
corr_starwars = pd.DataFrame(smilar_to_starwars, columns=["Correlation"])
corr_starwars.dropna(inplace=True)
corr_starwars.head()

# Elde ettiğimiz dataframe'i sıralayım ve görelim bakalım star Wars'a en yakın tavsiye edeceği film neymiş:
corr_starwars.sort_values("Correlation", ascending=False).head(10)

# Görüldüğü gibi alakasız sonuçlar çıktı, bu konuyu biraz araştırınca bunun nedeninin bu filmlerin çok az oy alması nedeniyle olduğunu bulacaksınız.. 
# Bu durumu düzeltmek için 100'den az oy alan filmleri eleyelim.. Bu amaçla ratings isimli bir dataframe oluşturalım ve burada her fimin kaç tane oy 
# aldığını (yani oy sayısını) tutalım...


df.head()

# timestamp sütununa ihtiyacımız yok silelim...
df.drop(["timestamp"], axis = 1)

# Her filmin ortalama (mean value) rating değerini bulalım 
ratings = pd.DataFrame(df.groupby("title")["rating"].mean())
# Büyükten küçüğe sıralayıp bakalım...
ratings.sort_values("rating", ascending = False).head()

# Dikkat: Bu ortalamalar hesaplanırken kaç oy aldığına bakmadık o yüzden böyle hiç bilmediğimiz filmler çıktı..
ratings["rating_oy_sayisi"] = pd.DataFrame(df.groupby("title")["rating"].count())
ratings.head()

# Şimdi en çok oy alan filmleri büyükten küçüğe sıralayıp bakalım...
ratings.sort_values("rating_oy_sayisi", ascending=False).head()

# Tekrar esas amacımıza dönelim ve corr_starwars dataframe'imize rating_oy_sayisi sütununu ekleyelim (join ile) 
corr_starwars.sort_values("Correlation", ascending=False).head()

corr_starwars = corr_starwars.join(ratings["rating_oy_sayisi"])
corr_starwars.head()

# ve sonuç
corr_starwars[corr_starwars["rating_oy_sayisi"] > 100].sort_values("Correlation", ascending = False).head()






