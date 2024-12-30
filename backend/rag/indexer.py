from typing import List, Dict, Any
import faiss
import numpy as np
from pathlib import Path
import json

class VectorIndex:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents: List[Dict[str, Any]] = []
        
    def add(self, embeddings: np.ndarray, documents: List[Dict[str, Any]]):
        self.index.add(embeddings)
        self.documents.extend(documents)
        
    def search(self, query_embedding: np.ndarray, k: int = 5):
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        return [self.documents[i] for i in indices[0]]
    
    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "documents.json", "w") as f:
            json.dump(self.documents, f)
            
    @classmethod
    def load(cls, path: Path):
        index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "documents.json", "r") as f:
            documents = json.load(f)
        
        instance = cls(index.d)
        instance.index = index
        instance.documents = documents
        return instance 