# fastapi_app/app.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os
from fastapi_app.model_utils import LayoutRecommender

UPLOAD_DIR = 'uploads'
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(title="ML Layout Recommender API")

# Initialize recommender
recommender = LayoutRecommender(
    embedding_model_dir='models/embedding_extractor',
    index_path='models/gallery.index',
    gallery_paths_path='models/gallery_paths.npy'
)

@app.post("/recommend")
async def recommend(file: UploadFile = File(...), top_k: int = 5):
    """Upload an image and get top-k recommended layouts"""
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Run recommendation
    recommendations = recommender.recommend(file_path, top_k=top_k)
    
    return JSONResponse(content={"query_image": file.filename, "recommendations": recommendations})
