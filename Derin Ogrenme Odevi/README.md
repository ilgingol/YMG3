# Derin Öğrenme Ödevi

**Ilgın Göl · 02220224035**

Fashion-MNIST kıyafet görsellerini sınıflandırmak için TensorFlow ve Keras ile iki CNN modeli eğitildi. Deneylerde yalnızca ilk konvolüsyon katmanının filtre sayısı değiştirildi.

[Word raporu](Derin%20%C3%96%C4%9Frenme%20%C3%96devi.docx)  · [Python kodu](main.py)

## Deney sonuçları

| Ölçüt | 8 filtre | 16 filtre |
| --- | ---: | ---: |
| Eğitim doğruluğu, son epoch | %88,32 | %88,72 |
| Doğrulama doğruluğu, son epoch | %87,53 | %87,72 |
| Test doğruluğu | %87,19 | %87,50 |
| Test kaybı | 0,3623 | 0,3545 |
| Eğitim süresi | 20,57 sn | 21,53 sn |

İki model de 5 epoch boyunca eğitildi ve 10.000 test görseliyle değerlendirildi. Süre, epoch sonu doğrulamayı da kapsar. [Sayısal çıktılar](ciktilar/04_test_ve_karsilastirma.txt) · [Ham ölçümler](ciktilar/deney_sonuclari.json)

## Teslim dosyaları

| Yönergedeki madde | Dosya veya bölüm |
| --- | --- |
| 1. Teorik sorular ve rapor | [Word raporunda](Derin%20%C3%96%C4%9Frenme%20%C3%96devi.docx) beş soru grubu ve model mimarisi soruları |
| 2. Python uygulaması ve ders sunumu | [main.py](main.py), [requirements.txt](requirements.txt) ve [PowerPoint sunumu](Derin%20%C3%96%C4%9Frenme%20Sunumu.pptx) |
| 3. Konvolüsyon filtreleri | [16 filtrenin başlangıç ağırlıkları](ciktilar/01_baslangic_filtreleri.png) |
| 4. Max Pooling giriş ve çıkışı | [Giriş ve çıkış görseli](ciktilar/02_max_pooling.png) ve Word raporunda elle hesaplama |
| 5. Eğitim ve doğrulama grafikleri | [Birleşik grafik](ciktilar/03_egitim_ve_dogrulama.png), [8 filtreli model](ciktilar/03_egitim_ve_dogrulama_8_filtre.png), [16 filtreli model](ciktilar/03_egitim_ve_dogrulama_16_filtre.png) |
| 6. Modelin test sonuçları | [Test doğruluğu ve kaybı](ciktilar/04_test_ve_karsilastirma.txt) |
| 7. İki deneyin karşılaştırması | Yukarıdaki tablo ve [ham ölçümler](ciktilar/deney_sonuclari.json) |
| 8. Değerlendirme ve sonuç | Word raporunun **Kendi değerlendirmem ve sonuç** bölümü ile sunumun son slaytı |

## Uygulamayı çalıştırma

Windows’ta Python 3.13 ile, proje klasöründe şu PowerShell komutlarını çalıştırın.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

İlk çalıştırmada Keras Fashion-MNIST’i otomatik indirir. Program yeni grafik ve ölçümleri `ciktilar/` klasörüne yazar. Eğitim süresi cihaz ve çalıştırma koşullarına göre değişebilir.
