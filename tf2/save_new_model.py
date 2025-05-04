# save_new_model.py
import tensorflow as tf

def simclr_augment(image):
    image = tf.image.random_crop(image, size=(32, 32, 3))
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.8)
    image = tf.image.random_contrast(image, 0.6, 1.4)
    image = tf.image.random_hue(image, 0.4)
    return tf.clip_by_value(image, 0.0, 1.0)

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

# Model oluştur
model = create_simclr_model()

# Modeli kaydet
model.save('./simclr_model.h5')
print("Model kaydedildi.")