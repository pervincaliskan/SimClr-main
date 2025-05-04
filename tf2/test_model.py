import tensorflow as tf
import os
import ssl  # SSL için

# SSL doğrulamasını atla
ssl._create_default_https_context = ssl._create_unverified_context

# İlk olarak model var mı kontrol edin
model_path = './simclr_model.h5'

if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    print("Model başarıyla yüklendi!")

    # Test verileri yükle
    (_, _), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    test_images = test_images.astype('float32') / 255.0

    # İlk 10 test görüntüsü için encoder çıktısını al
    test_batch = test_images[:10]
    embeddings, _ = model.predict(test_batch)

    print("Embeddings shape:", embeddings.shape)
    print("First embedding example:", embeddings[0][:10])
else:
    print(f"Model dosyası bulunamadı: {model_path}")
    print("Önce modeli oluşturmanız ve kaydetmeniz gerekiyor.")

    # Yeni bir model oluşturalım
    import tensorflow as tf

    def create_simclr_model():
        inputs = tf.keras.Input(shape=(32, 32, 3))
        x = tf.keras.layers.Conv2D(64, 3, activation='relu')(inputs)
        x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
        h = tf.keras.layers.GlobalAveragePooling2D()(x)
        z = tf.keras.layers.Dense(128, activation='relu')(h)
        z = tf.keras.layers.Dense(64)(z)
        return tf.keras.Model(inputs=[inputs], outputs=[h, z])

    model = create_simclr_model()
    model.save(model_path)
    print(f"Yeni model oluşturuldu ve kaydedildi: {model_path}")