# fastapi_app/model_utils.py
import os
import numpy as np
import tensorflow as tf
import faiss
from tensorflow.keras.preprocessing import image

IMG_SIZE = (224,224)

class LayoutRecommender:
    def __init__(self, embedding_model_dir, index_path, gallery_paths_path):
        # Load embedding model
        self.model = tf.keras.models.load_model(embedding_model_dir)
        # Load Faiss index
        self.index = faiss.read_index(index_path)
        # Load gallery image paths
        self.gallery_paths = np.load(gallery_paths_path)

    def preprocess_image(self, img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        return np.expand_dims(img.numpy(), axis=0)

    def recommend(self, query_img_path, top_k=5):
        """Return top-k recommended gallery image paths for a query image"""
        img_tensor = self.preprocess_image(query_img_path)
        emb = self.model.predict(img_tensor)
        # Faiss search
        D, I = self.index.search(emb.astype('float32'), top_k)
        recommended_paths = [self.gallery_paths[i] for i in I[0]]
        return recommended_paths
