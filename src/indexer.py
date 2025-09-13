# src/indexer.py
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import faiss  # pip install faiss-cpu
from src.models import build_embedding_model

BATCH_SIZE = 32
IMG_SIZE = (224,224)

def preprocess_image(path):
    """Load and preprocess a single image for embedding extraction"""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img

def extract_embeddings(image_paths, embedding_model, batch_size=BATCH_SIZE):
    """Extract embeddings for a list of images"""
    embeddings = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_imgs = np.stack([preprocess_image(p).numpy() for p in batch_paths])
        emb = embedding_model.predict(batch_imgs)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings)
    return embeddings

def build_faiss_index(embeddings, save_dir='models'):
    """Build and save a Faiss index from embeddings"""
    os.makedirs(save_dir, exist_ok=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype('float32'))
    faiss.write_index(index, os.path.join(save_dir, 'gallery.index'))
    print(f"Faiss index saved to {save_dir}/gallery.index")
    return index

def save_mapping(image_paths, save_dir='models'):
    """Save image path mapping for retrieval"""
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'gallery_paths.npy'), np.array(image_paths))
    print(f"Image path mapping saved to {save_dir}/gallery_paths.npy")

def main(data_csv, embedding_model_dir='models/embedding_extractor', save_dir='models'):
    # Load gallery image paths
    df = pd.read_csv(data_csv)
    image_paths = df['image_path'].values

    # Load embedding model
    embedding_model = tf.keras.models.load_model(embedding_model_dir)

    # Extract embeddings
    embeddings = extract_embeddings(image_paths, embedding_model)
    print(f"Extracted embeddings for {len(image_paths)} images.")

    # Build Faiss index
    build_faiss_index(embeddings, save_dir)

    # Save mapping
    save_mapping(image_paths, save_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_csv', type=str, required=True, help='CSV with gallery image paths')
    parser.add_argument('--embedding_model_dir', type=str, default='models/embedding_extractor')
    parser.add_argument('--save_dir', type=str, default='models')
    args = parser.parse_args()

    main(args.data_csv, args.embedding_model_dir, args.save_dir)
