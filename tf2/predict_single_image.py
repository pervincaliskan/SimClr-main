import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import ssl
from sklearn.linear_model import LogisticRegression
import cv2

# SSL doğrulamasını atla
ssl._create_default_https_context = ssl._create_unverified_context

# CIFAR-10 sınıf isimleri
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']


def predict_single_image(image_path, model, classifier):
    """Tek bir görüntüyü sınıflandırır"""

    # Görüntüyü yükle ve CIFAR-10 boyutuna getir
    if not os.path.exists(image_path):
        print(f"Hata: {image_path} dosyası bulunamadı!")
        return None

    # OpenCV ile yükle (BGR formatında)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Hata: {image_path} yüklenemedi!")
        return None

    # BGR'den RGB'ye çevir
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # CIFAR-10 boyutuna (32x32) getir
    img_resized = cv2.resize(img, (32, 32))

    # Normalize et
    img_normalized = img_resized.astype('float32') / 255.0

    # Batch boyutuna getir
    img_batch = np.expand_dims(img_normalized, axis=0)

    # Feature çıkar
    embeddings, _ = model.predict(img_batch, verbose=0)

    # Tahmin yap
    prediction = classifier.predict(embeddings)[0]
    confidence = classifier.predict_proba(embeddings)[0]

    # Görselleştir
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Orijinal resmi göster
    ax1.imshow(img)
    ax1.set_title('Orijinal Resim', fontsize=16)
    ax1.axis('off')

    # 32x32 versiyonu göster
    ax2.imshow(img_resized)
    ax2.set_title(f'Tahmin: {CIFAR10_CLASSES[prediction]}\n'
                  f'Güven: {confidence[prediction] * 100:.1f}%',
                  fontsize=16, color='green' if confidence[prediction] > 0.5 else 'red')
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig('single_prediction.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Güven oranlarını göster
    plt.figure(figsize=(12, 8))
    bars = plt.bar(CIFAR10_CLASSES, confidence * 100)

    # En yüksek güveni farklı renkte göster
    bars[prediction].set_color('green')

    plt.title('Tahmin Güven Oranları', fontsize=16)
    plt.xlabel('Sınıf', fontsize=12)
    plt.ylabel('Güven (%)', fontsize=12)
    plt.xticks(rotation=45)
    plt.ylim(0, 100)

    # Değerleri bar'ların üstüne yaz
    for bar, conf in zip(bars, confidence):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{conf * 100:.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('confidence_scores.png', dpi=300, bbox_inches='tight')
    plt.close()

    return prediction, confidence


def test_with_cifar_examples(model, classifier, n_examples=9):
    """CIFAR-10'dan rastgele köpek resimleri test et"""

    # CIFAR-10 yükle
    (_, _), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    test_images = test_images.astype('float32') / 255.0
    test_labels = test_labels.ravel()

    # Köpek örneklerini bul (index: 5)
    dog_indices = np.where(test_labels == 5)[0]

    # Rastgele köpek resimleri seç
    selected_indices = np.random.choice(dog_indices, n_examples, replace=False)

    # Görselleştir
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.ravel()

    for i, idx in enumerate(selected_indices):
        img = test_images[idx]

        # Tahmin yap
        img_batch = np.expand_dims(img, axis=0)
        embeddings, _ = model.predict(img_batch, verbose=0)
        prediction = classifier.predict(embeddings)[0]
        confidence = classifier.predict_proba(embeddings)[0][prediction]

        # Görselleştir
        axes[i].imshow(img)
        is_correct = (prediction == 5)  # 5 = dog
        color = 'green' if is_correct else 'red'

        axes[i].set_title(f'True: {CIFAR10_CLASSES[5]}\n'
                          f'Pred: {CIFAR10_CLASSES[prediction]}\n'
                          f'Conf: {confidence * 100:.1f}%',
                          color=color, fontsize=12, fontweight='bold')
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('dog_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    model_path = './simclr_model.h5'

    if not os.path.exists(model_path):
        print("Model bulunamadı. Önce simclr_basic.py'yi çalıştırın.")
        return

    # Model yükle
    print("Model yükleniyor...")
    model = tf.keras.models.load_model(model_path)

    # Classifier eğit
    print("Classifier eğitiliyor...")
    (train_images, train_labels), _ = tf.keras.datasets.cifar10.load_data()
    train_images = train_images.astype('float32') / 255.0
    train_labels = train_labels.ravel()

    train_embeddings, _ = model.predict(train_images, verbose=0)
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(train_embeddings, train_labels)

    # Kendi resminizi test edin!
    image_path = input("Test etmek istediğiniz resmin yolunu girin (örn: kopek.jpg): ")

    if image_path:
        prediction, confidence = predict_single_image(image_path, model, classifier)

        if prediction is not None:
            print(f"\nTahmin: {CIFAR10_CLASSES[prediction]}")
            print(f"Güven: {confidence[prediction] * 100:.1f}%")
            print("\nSonuç görselleri kaydedildi:")
            print("- single_prediction.png")
            print("- confidence_scores.png")

    # CIFAR-10'dan köpek resimleri test et
    print("\nCIFAR-10'dan köpek resimleri test ediliyor...")
    test_with_cifar_examples(model, classifier)
    print("Köpek tahminleri kaydedildi: dog_predictions.png")


if __name__ == "__main__":
    main()