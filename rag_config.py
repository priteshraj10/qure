from dataclasses import dataclass
from typing import Optional, List

@dataclass
class RAGConfig:
    # Embedding model settings
    embedding_model: str = "pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb"
    embedding_dim: int = 768
    
    # Retrieval settings
    max_chunks_per_doc: int = 10
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    # Vector store settings
    vector_store_path: str = "vector_store"
    
    # Reranking settings
    use_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    reranker_threshold: float = 0.5
    
    # RAG settings
    max_retrieved_documents: int = 5
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Image processing settings
    image_size: tuple = (224, 224)
    image_mean: tuple = (0.485, 0.456, 0.406)
    image_std: tuple = (0.229, 0.224, 0.225)
    
    # Cache settings
    use_cache: bool = True
    cache_dir: str = ".cache"
    
    # Prompt templates
    system_prompt: str = """You are a medical AI assistant. Given a medical image and/or text query, 
    provide accurate and relevant information based on the retrieved medical literature. 
    Always cite your sources and maintain a professional, scientific tone."""
    
    query_prompt: str = """Context: {context}
    
    Image Description: {image_description}
    
    Question: {question}
    
    Based on the provided context and image (if any), please provide a detailed answer:"""
    
    # Logging settings
    log_level: str = "INFO"
    wandb_project: str = "medical-rag"
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items()} 