import tensorflow as tf
import ssl
import numpy as np

# SSL doğrulamasını atla
ssl._create_default_https_context = ssl._create_unverified_context


# Veri augmentasyonu
def simclr_augment(image):
    image = tf.image.random_crop(image, size=(32, 32, 3))
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.8)
    image = tf.image.random_contrast(image, 0.6, 1.4)
    image = tf.image.random_hue(image, 0.4)
    return tf.clip_by_value(image, 0.0, 1.0)


# Model oluştur
def create_simclr_model():
    inputs = tf.keras.Input(shape=(32, 32, 3))

    # Encoder (ResNet)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
    h = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Projeksiyon başlığı
    z = tf.keras.layers.Dense(128, activation='relu')(h)
    z = tf.keras.layers.Dense(64)(z)

    return tf.keras.Model(inputs=[inputs], outputs=[h, z])


# Contrastive Loss
def contrastive_loss(z1, z2, temperature=0.5):
    # L2 normalize
    z1 = tf.math.l2_normalize(z1, axis=1)
    z2 = tf.math.l2_normalize(z2, axis=1)

    # Benzerlik matrisi
    similarity = tf.matmul(z1, z2, transpose_b=True) / temperature

    batch_size = tf.shape(z1)[0]
    labels = tf.range(batch_size)

    loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=similarity))
    loss2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=tf.transpose(similarity)))

    return (loss1 + loss2) / 2.0


# Eğitim adımı
@tf.function
def train_step(model, optimizer, batch, temperature):
    # İki augment versiyonu oluştur
    aug1 = tf.map_fn(simclr_augment, batch)
    aug2 = tf.map_fn(simclr_augment, batch)

    with tf.GradientTape() as tape:
        # Forward pass
        h1, z1 = model(aug1)
        h2, z2 = model(aug2)

        # Loss hesapla
        loss = contrastive_loss(z1, z2, temperature)

    # Gradyan güncelle
    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return loss


# Ana eğitim döngüsü
def main():
    model = create_simclr_model()
    optimizer = tf.keras.optimizers.Adam(1e-3)
    temperature = 0.5

    # CIFAR-10 yükle
    print("Loading CIFAR-10 dataset...")
    (train_images, _), _ = tf.keras.datasets.cifar10.load_data()
    train_images = train_images.astype('float32') / 255.0

    dataset = tf.data.Dataset.from_tensor_slices(train_images)
    dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.batch(32)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # Loss değerlerini sakla
    loss_history = []

    # Eğitim
    print("Starting training...")
    for epoch in range(10):
        for i, batch in enumerate(dataset):
            loss = train_step(model, optimizer, batch, temperature)
            loss_history.append(loss.numpy())

            if i % 100 == 0:
                print(f"Epoch {epoch}, Step {i}, Loss: {loss:.4f}")

    print("Training completed!")

    # Model kaydet
    model_save_path = './simclr_model.h5'
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    # Loss değerlerini dosyaya kaydet
    np.save('loss_history.npy', loss_history)
    print(f"Loss history saved to loss_history.npy")


if __name__ == "__main__":
    main()