### **`README.md` (initial)**

```markdown
# ML Layout Recommender (TensorFlow/Keras)

**Version:** 0.1.0

---

## Overview
This project is an **ML-based layout recommendation system** built using **TensorFlow/Keras**.  
It takes an input layout image and recommends the top-k visually similar layouts from a gallery using **deep embeddings** and **Faiss** for fast similarity search.

---

## Features
- Uses a pretrained backbone (EfficientNet) to extract embeddings from images.
- Embeddings are **L2-normalized** for similarity search.
- Builds a **Faiss index** for fast nearest-neighbor retrieval.
- Provides a **FastAPI** endpoint to query top-k layout recommendations.
- Includes a **Colab-friendly demo notebook** (`demo.ipynb`) to test the pipeline end-to-end.
- Automatic download of sample images from TensorFlow datasets (no manual download required).

---

## Project Structure
```

ml\_layout\_recommender\_tf/
├─ src/
│  ├─ models.py        # Embedding model definition
│  ├─ train.py         # Training script
│  ├─ indexer.py       # Embedding extraction + Faiss index creation
│  └─ utils.py         # Optional helper functions
├─ fastapi\_app/
│  ├─ app.py           # FastAPI server
│  └─ model\_utils.py   # Embedding & recommendation utilities
├─ data/               # Sample images (auto-downloaded in demo)
├─ notebooks/
│  └─ demo.ipynb       # Interactive Colab demo
├─ models/             # Trained models + Faiss index
├─ requirements.txt
└─ README.md

````

---

## Quick Start

### 1. Run Demo Notebook
Open `demo.ipynb` in **Google Colab** and run all cells to:
- Download sample images
- Build embedding model
- Precompute embeddings
- Build Faiss index
- Query sample images and visualize top-k recommendations

### 2. Run FastAPI (optional)
```bash
cd fastapi_app
uvicorn app:app --reload
````

* Open `http://127.0.0.1:8000/docs`
* Use the `/recommend` endpoint to upload an image and get top-k recommended layouts

---

## Dependencies

* Python 3.8+
* TensorFlow 2.x
* Faiss (`faiss-cpu`)
* FastAPI
* Pillow
* Pandas
* Matplotlib

Install via:

```bash
pip install -r requirements.txt
```

---

## License

This project is released under the MIT License.


