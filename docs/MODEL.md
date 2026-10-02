# Model ve eğitim bilgisi

- Temel VLM: https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct
- Veri: https://github.com/CAMMA-public/cholect50
- Mimari: VLM'nin kendi görüntü kodlayıcısı + bağlantı katmanı + dil modeli. Ayrı ResNet/YOLO yok.
- Uyarlama: Dil modelinin q_proj/v_proj katmanlarında rank 8 LoRA; connector da eğitildi.
- Eğitim: 3 epoch; her epoch 6.000 benzersiz, nadir sınıflara ağırlık veren kare.
- Eğitim havuzu: Resmi challenge ayrımındaki 40 video; doğrulama 5 ayrı video.
- Mevcut eğitim havuzundaki 97 üçlü sınıfı örneklemeye dahil edildi.
- Seçim: Sabit 100 doğrulama karesinde üçlü micro/macro F1 ortalaması.
- Paylaşılan adaptör: Üçüncü epoch, en yüksek seçim skoru.

Mevcut modelin 100 doğrulama görüntüsündeki sonuçları:

| Görev | F1 |
|---|---:|
| Alet | %77,5 |
| Alet–eylem–hedef | %50,5 |
| Hedef (üçlülerden) | %64,3 |
| Eylem (üçlülerden) | %72,2 |

Bu değerler doğrulama sonuçlarıdır, bağımsız tam test sonucu veya resmi benchmark değildir.
Model bu örneklerle seçildi. 100 kare bütün koşulları/sınıfları kapsamaz.
Alet tanıma, doğru eylem-hedef eşleşmesinden daha başarılıdır. Yanlış ve eksik tahminler olabilir.
Alet listesi ve üçlüler bazen birbiriyle tutarsız olabilir; ham çıktılar inceleme için kaydedilir.

## Kapsam ve kaynak koşulları

Bu sürüm araştırma/eğitim prototipidir; cerrahi karar desteği olarak doğrulanmamıştır.
CholecT50 bir genel cerrahi veri seti değildir; laparoskopik safra kesesi ameliyatlarına odaklanır.
Genel tıbbi amaç açıklaması için kaynaklı bir bilgi tabanı bağlı değildir.
CholecT50'nin kendi lisansı CC BY-NC-SA 4.0, temel SmolVLM modelinin lisansı Apache-2.0'dır.
Kaynaklardaki kullanım/dağıtım koşulları ayrıca geçerlidir; bu README yeni bir lisans tanımlamaz.
Veri seti görüntüleri bu dağıtıma eklenmemiştir.

## Geliştirici için

Proje hem tahmin hem isteğe bağlı yeniden eğitim içerir. train.py, training/ içindeki
yardımcıları kullanır. Proje bağımsız çalışır.
Yeni eğitim hazır model/ klasörünü değiştirmez; --adapter ile yeni modeli seçin.
