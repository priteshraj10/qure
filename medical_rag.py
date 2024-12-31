import torch
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from PIL import Image
import faiss
import numpy as np
from typing import List, Dict, Optional, Union
import logging
from pathlib import Path
from rag_config import RAGConfig
from sentence_transformers import CrossEncoder
from torch.utils.data import Dataset, DataLoader

class MedicalRAG:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.setup_logging()
        self.setup_models()
        self.setup_vector_store()
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_models(self):
        """Initialize all required models"""
        # Load embedding model
        self.embedding_model = AutoModel.from_pretrained(self.config.embedding_model)
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(self.config.embedding_model)
        
        # Load reranker if enabled
        if self.config.use_reranker:
            self.reranker = CrossEncoder(self.config.reranker_model)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model.to(self.device)
    
    def setup_vector_store(self):
        """Initialize FAISS vector store"""
        self.index = faiss.IndexFlatL2(self.config.embedding_dim)
        self.document_store = []
        
        # Load existing index if available
        if Path(self.config.vector_store_path).exists():
            self.load_vector_store()
    
    def process_document(self, text: str) -> List[str]:
        """Split document into chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.config.chunk_size - self.config.chunk_overlap):
            chunk = " ".join(words[i:i + self.config.chunk_size])
            chunks.append(chunk)
            
            if len(chunks) >= self.config.max_chunks_per_doc:
                break
        
        return chunks
    
    def get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Get embeddings for text chunks"""
        embeddings = []
        
        for text in texts:
            inputs = self.embedding_tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1)
                embeddings.append(embedding)
        
        return torch.cat(embeddings, dim=0)
    
    def add_to_index(self, documents: List[Dict[str, Union[str, Image.Image]]]):
        """Add documents to the vector store"""
        for doc in documents:
            # Process text
            chunks = self.process_document(doc["text"])
            embeddings = self.get_embeddings(chunks)
            
            # Add to FAISS index
            self.index.add(embeddings.cpu().numpy())
            
            # Store document info
            for chunk in chunks:
                self.document_store.append({
                    "text": chunk,
                    "source": doc.get("source", "unknown"),
                    "image_path": doc.get("image_path", None)
                })
    
    def retrieve(self, query: str, k: int = None) -> List[Dict]:
        """Retrieve relevant documents"""
        if k is None:
            k = self.config.max_retrieved_documents
        
        # Get query embedding
        query_embedding = self.get_embeddings([query])
        
        # Search in FAISS
        distances, indices = self.index.search(query_embedding.cpu().numpy(), k)
        
        # Get retrieved documents
        retrieved_docs = [self.document_store[i] for i in indices[0]]
        
        # Rerank if enabled
        if self.config.use_reranker:
            reranked_docs = self.rerank(query, retrieved_docs)
            return reranked_docs
        
        return retrieved_docs
    
    def rerank(self, query: str, documents: List[Dict]) -> List[Dict]:
        """Rerank retrieved documents"""
        pairs = [(query, doc["text"]) for doc in documents]
        scores = self.reranker.predict(pairs)
        
        # Filter and sort by score
        scored_docs = [
            (doc, score) for doc, score in zip(documents, scores)
            if score >= self.config.reranker_threshold
        ]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in scored_docs]
    
    def process_query(self, query: str, image: Optional[Image.Image] = None) -> str:
        """Process a query with optional image"""
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(query)
        
        # Prepare context
        context = "\n\n".join([f"Source {i+1}: {doc['text']}" 
                             for i, doc in enumerate(retrieved_docs)])
        
        # Process image if provided
        image_description = ""
        if image:
            # TODO: Implement image captioning or feature extraction
            pass
        
        # Format prompt
        prompt = self.config.query_prompt.format(
            context=context,
            image_description=image_description,
            question=query
        )
        
        return prompt
    
    def save_vector_store(self):
        """Save vector store and document index"""
        path = Path(self.config.vector_store_path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path / "index.faiss"))
        
        # Save document store
        import json
        with open(path / "documents.json", "w") as f:
            json.dump(self.document_store, f)
    
    def load_vector_store(self):
        """Load vector store and document index"""
        path = Path(self.config.vector_store_path)
        
        # Load FAISS index
        self.index = faiss.read_index(str(path / "index.faiss"))
        
        # Load document store
        import json
        with open(path / "documents.json", "r") as f:
            self.document_store = json.load(f)

class PubMedVisionDataset(Dataset):
    """Dataset class for PubMedVision"""
    def __init__(self, dataset_path: str, config: RAGConfig):
        self.config = config
        self.data = self.load_dataset(dataset_path)
    
    def load_dataset(self, path: str):
        """Load and preprocess the dataset"""
        from datasets import load_dataset
        dataset = load_dataset("FreedomIntelligence/PubMedVision")
        return dataset
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        
        return {
            "image": image,
            "text": item["text"],
            "caption": item.get("caption", "")
        } 