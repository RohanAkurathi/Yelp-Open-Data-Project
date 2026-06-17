"""FAISS index helpers.

Embeddings are L2-normalised, so an inner-product index gives cosine similarity.
At our corpus size a flat (exact) index is fast and exact; the same interface
swaps to an IVF/HNSW index unchanged when the corpus grows.
"""

import faiss
import numpy as np

from src import config as C

# single-threaded FAISS avoids native thread-pool clashes with torch/OpenBLAS
# when both live in one process (e.g. the Streamlit app)
faiss.omp_set_num_threads(1)

INDEX_PATH = C.ARTIFACTS / "faiss.index"


def build_index(embeddings: np.ndarray) -> faiss.Index:
    emb = np.ascontiguousarray(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index


def save_index(index: faiss.Index, path=INDEX_PATH) -> None:
    faiss.write_index(index, str(path))


def load_index(path=INDEX_PATH) -> faiss.Index:
    return faiss.read_index(str(path))


def search(index: faiss.Index, query_emb: np.ndarray, k: int):
    q = np.ascontiguousarray(query_emb, dtype="float32")
    if q.ndim == 1:
        q = q[None, :]
    scores, idx = index.search(q, k)
    return scores, idx
