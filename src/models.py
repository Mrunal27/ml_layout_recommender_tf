import tensorflow as tf
from tensorflow.keras import layers

def build_embedding_model(input_shape=(224, 224, 3), embedding_dim=128, backbone_name='EfficientNetB0'):

    base_model = getattr(tf.keras.applications, backbone_name)(
        include_top = False,
        weights = 'imagenet',
        input_shape = input_shape
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape = input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(embedding_dim)(x)

    outputs = layers.Lambda(lambda t:tf.math.l2_normalize(t, axis=1))(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='embedding_model')
    return model

if __name__ == "__main__":
    model = build_embedding_model()
    model.summary()