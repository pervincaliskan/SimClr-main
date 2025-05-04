# SimCLR Türkçe Uygulaması

CIFAR-10 veri seti için TensorFlow 2.x kullanılarak geliştirilmiş basit SimCLR implementasyonu.


## Genel Bakış

SimCLR (Simple Framework for Contrastive Learning), görsel temsilleri öğrenmek için kendinden denetimli öğrenme kullanan bir framework'tür. Bu proje, algoritmanın basit bir implementasyonunu içerir.

## Özellikler

- CIFAR-10 veri seti eğitimi
- Contrastive learning ile özellik öğrenimi
- Loss değerlerinin görselleştirilmesi
- Eğitilmiş modeli kaydetme

## Gereksinimler

```bash
Python 3.8+
TensorFlow 2.x
NumPy
Matplotlib
```

## Kurulum

```bash
# TensorFlow ve bağımlılıkları yükle
pip install tensorflow numpy matplotlib

# SSL sertifikaları (macOS için)
/Applications/Python\ 3.8/Install\ Certificates.command
```

## Kullanım

### Modeli Eğitmek

```bash
python simclr_basic.py
```

Bu komut:
- CIFAR-10 veri setini indirir
- Modeli 10 epoch eğitir
- `simclr_model.h5` olarak kaydeder
- Loss geçmişini `loss_history.npy` olarak kaydeder

### Loss Grafiğini Oluşturmak

```bash
python visualize_loss.py
```

Bu komut:
- Kaydedilen loss değerlerini okur
- Loss grafiğini oluşturur
- `loss_plot.png` olarak kaydeder

## Dosya Yapısı

```
├── simclr_basic.py         # Ana eğitim dosyası
├── visualize_loss.py       # Loss görselleştirme
├── loss_history.npy        # Loss değerleri
├── simclr_model.h5         # Eğitilmiş model
├── loss_plot.png           # Loss grafiği
└── README.md               # Bu dosya
```

## Algoritma Detayları

### Bileşenler

1. **Veri Augmentasyonu**
   - Random crop ve resize
   - Random color distortion

2. **Model Mimarisi**
   - Encoder: ResNet-50
   - Projection Head: 1-layer MLP

3. **Loss Fonksiyonu**
   - NT-Xent (Normalized Temperature-scaled Cross Entropy)

### Eğitim Parametreleri

| Parametre | Değer |
|-----------|-------|
| Batch Size | 256 |
| Epochs | 10 |
| Optimizer | LARS |
| Learning Rate | 4.8 |

## Sonuçlar

- Starting Loss: 3.42
- Final Loss: 2.31
- Eğitim Süresi: ~28 dakika

## Referans

Google Research SimCLR makalesi:
```
@inproceedings{chen2020simple,
  title={A Simple Framework for Contrastive Learning of Visual Representations},
  author={Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
  year={2020}
}
```

## Lisans

MIT License

## İletişim

Sorularınız için issue açabilirsiniz.
