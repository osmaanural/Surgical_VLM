# Cerrahi VLM

Ameliyat görüntülerindeki **aletleri, eylemleri ve hedef yapıları** tahmin eder;
sonuçları Türkçe açıklamalarla görüntünün yanında gösterir.
SmolVLM-500M, CholecT50 verisiyle LoRA ve görsel bağlantı katmanı eğitimi kullanılarak uyarlanmıştır.

**Eğitilmiş model `model/` içinde hazırdır. Tahmin yapmak için tekrar eğitim veya
CholecT50 indirmek gerekmez. Kendi görüntülerinizi kullanabilirsiniz.**

## Kurulum

Projeyi indirin ve proje klasöründe terminal açın. Python 3.11–3.13 kullanın.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Bu komutlar macOS/Linux içindir. Windows ve diğer seçenekler:
[Ayrıntılı kullanım](docs/KULLANIM.md).

Temel [SmolVLM-500M](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct)
ağırlıkları ilk çalıştırmada internetten indirilir. Eğitilmiş adaptör `model/`
klasöründen yüklenir. Görüntüler yerel bilgisayarda işlenir.

## Tahmin ve test

**Önce `test/` klasörüne kendi PNG veya JPG görüntülerinizi koyun.**
Bu depoda veri seti görüntüleri ve örnek ameliyat fotoğrafları paylaşılmaz;
yeni indirilen projede `test/` görüntü açısından boştur.

Tek görüntü:
```bash
python predict.py
```
`test/ornek.png` varsa onu, yoksa klasördeki ilk görüntüyü kullanır.
Belirli bir görüntü: `python predict.py --image test/fotograf.png`
Buradaki `fotograf.png` yerine kendi dosyanızın adını yazın.

Klasördeki bütün görüntüler:
```bash
python test.py
```

Türkçe görüntülü sonuçlar ve JSON çıktıları `results/` içine kaydedilir.
Etiketsiz görüntülerle bu komutlar yalnızca tahmin üretir; doğruluk skoru hesaplamaz.

## Veri seti — CholecT50

Veri seti **bu depoya dahil değildir**. Yalnızca yeniden eğitim veya gerçek etiketlerle
başarı ölçümü yapmak istiyorsanız ayrıca edinmeniz gerekir.

- [Resmî CholecT50 indirme ve erişim sayfası](https://github.com/CAMMA-public/cholect50/blob/master/docs/README-Downloads.md)
- [Resmî sayfada bağlantısı verilen CholecT50 erişim formu](https://forms.gle/GbMj8TwNoNpMUJuv9)
- [Veri seti ana deposu](https://github.com/CAMMA-public/cholect50)

Erişim için resmî sayfadaki talimatları ve başvuru formunu izleyin.
CholecT50 bölümünü kullanın; ayrı sunulan küçük Challenge Validation seti,
bu projenin tüm eğitim veri setinin yerine geçmez.

İndirdikten sonra arşivi proje klasörünün **dışına** çıkarın:
```text
calisma/
├── surgical_vlm/          # Bu proje
└── CholecT50/             # Ayrı indirilen veri seti
    ├── labels/
    │   ├── VID01.json
    │   └── ...
    └── videos/
        ├── VID01/
        │   ├── 000000.png
        │   └── ...
        └── ...
```

Veri seti kaynağı CC BY-NC-SA 4.0 lisansını belirtir. Kullanım koşulları ve istenen
atıflar için resmî sayfayı inceleyin. Veri setinin lisansı ile temel modelin lisansı farklıdır.

## İsteğe bağlı eğitim ve etiketli değerlendirme

Yukarıdaki klasör düzeninde, proje klasöründen yeni eğitim:
```bash
python train.py --data-dir ../CholecT50
```

Veri seti başka yerdeyse `--data-dir` değerini kendi dizininizle değiştirin.
Ayrı veri hazırlama komutu gerekmez. Mevcut `model/` değişmez;
yeni eğitim sonuçları `training_runs/` içine kaydedilir.

Doğrulama kümesinde precision/recall/F1 ölçümü:
```bash
python test.py --data-dir ../CholecT50 --split val --per-video 20
```
Model ve ayar seçimleri tamamlandığında son değerlendirme için `--split test` kullanın.

## Dosyalar

| Dosya / klasör | Amaç |
|---|---|
| `predict.py` | Tek görüntüde tahmin |
| `test.py` | Klasör tahmini veya etiketli değerlendirme |
| `train.py` | İsteğe bağlı yeni eğitim |
| `model/` | Hazır eğitilmiş adaptör ve işlemci dosyaları |
| `test/` | Kullanıcının kendi görüntüleri |
| `results/` | Oluşturulan tahmin sonuçları |
| `training/` | Eğitim ayarları ve yardımcı kodlar |

## GitHub'a dahil edilmeyenler

`.gitignore`; veri klasörlerini, test görüntülerini, sonuçları, eğitim çıktılarını,
sanal ortamları ve önbellekleri Git dışında tutar. Yerel dosyalar silinmez.
**Kodlar, `training/` ve eğitilmiş `model/` klasörü paylaşılır.**

Yayınlamadan önce `git status --short` ile gönderilecek dosyaları kontrol edin.
`.gitignore`, önceden takibe alınmış dosyaları otomatik çıkarmaz; ayrıca GitHub web
arayüzüne elle seçerek yüklenen dosyaları filtrelemez. Veri klasörlerini elle seçip yüklemeyin.

## Sınırlar

Çıktılar model tahminidir; hata yapabilir. Kutu, koordinat veya fiziksel ölçü üretilmez.
Bu çalışma araştırma prototipidir; cerrahi karar sistemi olarak doğrulanmamıştır.

[Ayrıntılı kullanım](docs/KULLANIM.md) · [Model bilgisi ve ölçümler](docs/MODEL.md)
