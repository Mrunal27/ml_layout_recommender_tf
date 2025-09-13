import os
import pandas as pd
import tensorflow as tf
from src.models import build_embedding_model
from tensorflow.keras import layers, models, optimizers, callbacks

AUTOTUNE = tf.data.AUTOTUNE
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EMBEDDING_DIM = 128

def preprocess_image(path, label=None):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    if label is not None:
        return img, label
    else:
        return img

def create_dataset(paths, labels=None, batch_size=BATCH_SIZE, shuffle=True):
    if labels is not None:
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(lambda p, l: preprocess_image(p, l), num_parallel_calls=AUTOTUNE)
    else:
        ds = tf.data.Dataset.from_tensor_slices(paths)
        ds = ds.map(lambda p: preprocess_image(p), num_parallel_calls = AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=2048)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds

def train_model(data_csv, output_dir='models', epochs=10):
    """
    Train the embedding model with optional classification head.
    data_csv: CSV file with columns 'image_path' and 'label'
    """
    df = pd.read_csv(data_csv)
    image_paths = df['image_path'].values
    labels = df['label'].values
    num_classes = len(set(labels))

    # Split into train/val
    val_split = 0.2
    val_size = int(len(image_paths) * val_split)
    train_paths, val_paths = image_paths[:-val_size], image_paths[-val_size:]
    train_labels, val_labels = labels[:-val_size], labels[-val_size:]

    # Datasets
    train_ds = create_dataset(train_paths, train_labels)
    val_ds = create_dataset(val_paths, val_labels, shuffle=False)

    # Build embedding model
    embedding_model = build_embedding_model(embedding_dim=EMBEDDING_DIM)

    # Attach classification head
    inputs = tf.keras.Input(shape=(224,224,3))
    x = embedding_model(inputs)
    logits = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=logits)

    model.compile(
        optimizer=optimizers.Adam(1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_cb = callbacks.ModelCheckpoint(
        os.path.join(output_dir, 'best_classif.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    earlystop_cb = callbacks.EarlyStopping(
        patience=5,
        monitor='val_loss',
        restore_best_weights=True
    )

    # Train
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[checkpoint_cb, earlystop_cb]
    )

    # Save embedding extractor only
    embedding_model.save(os.path.join(output_dir, 'embedding_extractor'))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_csv', type=str, required=True, help='CSV file with image_path,label')
    parser.add_argument('--output_dir', type=str, default='models')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    train_model(args.data_csv, args.output_dir, args.epochs)    
    