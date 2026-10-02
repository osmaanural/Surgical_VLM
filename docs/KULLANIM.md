# Cerrahi VLM — Eğitim, Tahmin ve Test

Görüntüdeki cerrahi aletleri ve alet–eylem–hedef ilişkilerini tahmin eder.
Türkçe açıklamaları görüntünün yanındaki panelde gösterir.
**Eğitilmiş model `model/` içinde hazırdır. Kullanmak için tekrar eğitim gerekmez.**

## Kurulum

Python 3.11–3.13. Kontrol edilen ortam: Python 3.13, Apple M4 / MPS.
Terminalde bu README'nin bulunduğu proje klasörünü açın.

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```
Windows PowerShell:
```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
Windows'ta aşağıdaki `python` yerine `.\.venv\Scripts\python.exe` kullanabilirsiniz.
NVIDIA CUDA destekli PyTorch kurulumu: https://pytorch.org/get-started/locally/
Windows/Linux/CUDA kod yolları mevcut, gerçek donanım kontrolü yalnızca Mac'te yapılmıştır.

## 1. Tek görüntüde tahmin — predict.py

Görüntünüzü `test/` içine koyun:
```bash
python predict.py --image test/fotograf.png
```
PNG görüntülü sonuç, Türkçe JSON ve ham yanıt `results/` altındaki yeni çalışma
klasörüne kaydedilir. Modelin varsayılan kaynağı `model/` klasörüdür.

## 2. Klasördeki bütün görüntüleri dene — test.py

Görüntülerinizi `test/` içine koyup:
```bash
python test.py
```
Alt klasörler de taranır. PNG, JPG, JPEG, WEBP, BMP desteklenir.
Başka bir klasör:
```bash
python test.py --input /gorseller/klasoru
```
Bu kullanım gerçek etiket gerektirmez ve doğruluk skoru hesaplamaz.
`python run.py` komutu da klasör tahmini için kullanılabilir.

### Etiketlerle gerçek başarı ölçümü

CholecT50'nin `labels/` ve `videos/` klasörlerini içeren ana dizinini verin:
```bash
python test.py --data-dir /veri/CholecT50 --split val --per-video 20
```
Beş doğrulama videosundan toplam 100 kare kullanır. Sonuçlar ve ham tahminler
`results/evaluation_.../evaluation.json` içine yazılır. Alet, eylem, hedef ve üçlü
precision/recall/F1 değerleri hesaplanır. Üçlü için üç bileşen birden eşleşmelidir.
Güven skoru bulunmadığından mAP hesaplanmaz. Geçersiz cevap boş tahmin sayılır;
geçersiz/bilinmeyen sınıf yanıtları ayrıca raporlanır.

Nihai test (model/ayar kararları tamamlandıktan sonra):
```bash
python test.py --data-dir /veri/CholecT50 --split test --per-video 20
```
`--per-video 0` bütün kareleri kullanır ve çok daha uzun sürebilir.
Bu sürümün skorlu testi kaldığı yerden devam etmez; rapor ölçüm tamamlandığında yazılır.
Varsayılan 100 kare bir örneklemdir, bütün veri setinin başarısı değildir.

## 3. Yeniden eğitim — train.py (isteğe bağlı)

Veri setini ayrıca edinmeniz gerekir; bu GitHub projesi görüntüleri içermez.
[Resmî indirme sayfası](https://github.com/CAMMA-public/cholect50/blob/master/docs/README-Downloads.md)
üzerindeki CholecT50 erişim formunu izleyin. Veri setini proje dışında tutun.
Kendi görüntülerinizle tahmin yapmak için veri setini indirmeniz gerekmez.
Beklenen yapı:
```text
CholecT50/
├── labels/VID01.json ...
└── videos/VID01/000000.png ...
```

```bash
python train.py --data-dir /veri/CholecT50
```
Sizin yerel projeniz CholecT50 altında duruyorsa kısayol:
```bash
python train.py --data-dir ..
```
Ayrı hazırlama komutu gerekmez. Kod etiketleri doğrudan okur, resmi video ayrımını uygular.
Eğitim verisine test videolarını katmaz. Ayarlar `training/config.json` içinde.
Varsayılan: 3 epoch, 6.000 farklı kare/epoch, rank 8 dil LoRA + görsel connector eğitimi.
Örnekleme nadir üçlü sınıflarını korur; tek kare/batch, 4 adım gradient accumulation.
En iyi model 100 sabit doğrulama karesindeki üçlü micro/macro F1 ortalamasıyla seçilir.
Başarı gelişmezse erken durdurulur. GPU bulunmazsa uzun CPU eğitimi başlatılmaz.

**Eğitim, temel SmolVLM'den yeni bir adaptör oluşturur.** Hazır `model/` üzerine yazmaz;
mevcut adaptörden kaldığı yerden devam etmez. Yeni çıktılar:
```text
training_runs/TARIH-SAAT/
├── best/          # Doğrulamada en iyi adaptör
├── last/          # Son epoch adaptörü
├── config.json
└── history.jsonl
```
Yeni modeli kullanmak için:
```bash
python predict.py --image test/fotograf.png --adapter training_runs/TARIH-SAAT/best
python test.py --adapter training_runs/TARIH-SAAT/best
```
TARIH-SAAT yerine oluşan gerçek klasör adını yazın. `model/` otomatik değiştirilmez.

Yalnızca veri ve bir ileri geçiş kontrolü (eğitim yapmaz):
```bash
python train.py --data-dir /veri/CholecT50 --check
```

## Proje yapısı

```text
surgical_vlm/
├── train.py          # Yeni eğitim
├── predict.py        # Tek görüntü tahmini
├── test.py           # Klasör testi / etiketli değerlendirme
├── run.py            # Ortak görüntülü tahmin akışı
├── model.py          # VLM yükleme
├── display.py        # Türkçe panel
├── model/            # Hazır eğitilmiş adaptör
├── training/         # Eğitim ayarları, veri okuma ve yardımcı kodlar
├── test/             # Kullanıcının görüntüleri
├── results/          # Tahmin/test çıktıları
├── requirements.txt
└── docs/MODEL.md
```

## GitHub ve taşınabilirlik

Bu klasörün içeriğini, **model/ ve training/ dahil**, paylaşın. `model/` içinde
LoRA/connector ağırlıkları ve işlemci dosyaları bulunur. Temel SmolVLM-500M ağırlıkları
ilk kullanımda indirilir. Sonraki kullanımlarda önbellekten okunur.
Görüntüler yerel bilgisayarda işlenir; bu kod görüntüleri bir API'ye yüklemez.
Mac MPS, CUDA veya CPU otomatik seçilir. Tahminde `--device cpu` kullanılabilir.
Dosya yolları taşınabilirdir; kullanıcı adına bağlı sabit dizinler kullanılmaz.

.gitignore sanal ortamları, kişisel `test/` görüntülerini, sonuçları ve eğitim deneylerini
Git dışında tutar. Veri setini `datasets/` altında tutarsanız o da hariç tutulur.
Veri setini projenin köküne kopyalamayın. GitHub'a otomatik yükleme yapılmamıştır.

## Sınırlar

Kutu, koordinat veya fiziksel ölçü üretilmez. Türkçe çıktılar sınıf sözlükleriyle çevrilir.
Çıktılar tahmindir; model hata yapabilir. Araştırma prototipidir, cerrahi karar için doğrulanmamıştır.
Kaynaklar, kullanım koşulları ve mevcut adaptörün ölçümleri: [Model bilgisi](docs/MODEL.md).
