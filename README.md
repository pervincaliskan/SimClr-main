SimCLR (Simple Framework for Contrastive Learning) Uygulaması
Bu proje, Google Research tarafından geliştirilen SimCLR (Simple Framework for Contrastive Learning of Visual Representations) algoritmasının basit bir implementasyonunu içerir.
Proje Hakkında
SimCLR, görsel temsillemleri öğrenmek için kendinden denetimli (self-supervised) öğrenme yaklaşımı kullanan bir framework'tür. Bu implementasyon, CIFAR-10 veri seti üzerinde çalışacak şekilde yapılandırılmıştır.
Ana Dosyalar

simclr_basic.py: SimCLR algoritmasının temel implementasyonu
visualize_loss.py: Eğitim sırasında elde edilen loss değerlerini görselleştiren yardımcı script
loss_history.npy: Eğitim sürecindeki loss değerlerinin kaydı
simclr_model.h5: Eğitilmiş model dosyası
loss_plot.png: Loss grafiği görselleştirmesi

Kurulum
Gereksinimler
bashPython 3.8+
TensorFlow 2.x
NumPy
Matplotlib
Ortam Kurulumu
bash# Gerekli paketlerin yüklenmesi
pip install tensorflow numpy matplotlib

# TensorFlow için sertifika güncelleme (macOS için)
/Applications/Python\ 3.8/Install\ Certificates.command
Kullanım
SimCLR Modelini Eğitme
bashpython simclr_basic.py
Bu komut:

CIFAR-10 veri setini indirir
SimCLR modelini 10 epoch boyunca eğitir
Model ağırlıklarını simclr_model.h5 olarak kaydeder
Loss değerlerini loss_history.npy olarak kaydeder

Loss Değerlerini Görselleştirme
bashpython visualize_loss.py
Bu komut:

Kaydedilen loss değerlerini yükler
Loss grafiğini görselleştirir
Grafiği loss_plot.png olarak kaydeder
Eğitim sürecine dair istatistikleri yazdırır

Algoritma Özellikleri
SimCLR Bileşenleri

Veri Augmentasyonu:

Random cropping ve resizing
Random color distortion
(Opsiyonel) Gaussian blur


Encoder (f):

ResNet-50 tabanlı
Görüntülerden feature vektörleri çıkarır


Projection Head (g):

1 katmanlı MLP
Feature'ları contrastive space'e map eder
ReLU aktivasyon fonksiyonu kullanır


Contrastive Loss:

NT-Xent (Normalized Temperature-scaled Cross Entropy)
Pozitif çiftleri birbirine yaklaştırır
Negatif çiftleri birbirinden uzaklaştırır



Eğitim Parametreleri

Batch Size: 256
Epochs: 10
Optimizer: LARS
Learning Rate: 4.8 (0.3 × BatchSize/256)
Temperature: 0.5

Sonuçlar
Eğitim Süreci

İlk loss değeri: ~3.42
Son loss değeri: ~2.31
Training süresi: ~28 dakika (10 epoch)
