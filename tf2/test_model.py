import tensorflow as tf
import numpy as np
import os
import ssl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# SSL doğrulamasını atla
ssl._create_default_https_context = ssl._create_unverified_context

# Model var mı kontrol et
model_path = './simclr_model.h5'

if os.path.exists(model_path):
    print("Model yükleniyor...")
    model = tf.keras.models.load_model(model_path)
    print("Model başarıyla yüklendi!\n")

    # CIFAR-10 veri setini yükle
    print("CIFAR-10 veri seti yükleniyor...")
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    # Veriyi normalize et
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0

    # Encoder çıktılarını al
    print("\nTrain veri seti için embedding'ler hesaplanıyor...")
    train_embeddings = []
    batch_size = 256  # Bellek için küçük batch kullan

    for i in range(0, len(train_images), batch_size):
        batch = train_images[i:i + batch_size]
        embeddings, _ = model.predict(batch, verbose=0)
        train_embeddings.extend(embeddings)

        # İlerleme göster
        if i % 2000 == 0:
            print(f"İşlenen: {i}/{len(train_images)}")

    train_embeddings = np.array(train_embeddings)
    train_labels = train_labels.ravel()

    print("\nTest veri seti için embedding'ler hesaplanıyor...")
    test_embeddings = []

    for i in range(0, len(test_images), batch_size):
        batch = test_images[i:i + batch_size]
        embeddings, _ = model.predict(batch, verbose=0)
        test_embeddings.extend(embeddings)

    test_embeddings = np.array(test_embeddings)
    test_labels = test_labels.ravel()

    # Linear classifier eğit
    print("\nLinear classifier eğitiliyor...")
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(train_embeddings, train_labels)

    # Test verileri üzerinde tahmin yap
    print("\nTahminler yapılıyor...")
    predictions = classifier.predict(test_embeddings)

    # Accuracy hesapla
    accuracy = accuracy_score(test_labels, predictions)

    # Sonuçları yazdır
    print("\n" + "=" * 50)
    print("SimCLR Evaluation Results:")
    print("=" * 50)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Total test samples: {len(test_labels)}")
    print(f"Correctly classified: {int(accuracy * len(test_labels))}")
    print("=" * 50)

else:
    print(f"Model dosyası bulunamadı: {model_path}")
    print("Önce simclr_basic.py dosyasını çalıştırarak modeli eğitmeniz gerekiyor.")