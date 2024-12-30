from typing import List, Dict, Any
from .embeddings import MedicalEmbeddings
from .indexer import VectorIndex
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class MedicalRAGPipeline:
    def __init__(self, config):
        self.config = config
        self.embeddings = MedicalEmbeddings()
        self.index = VectorIndex(dimension=768)  # PubMedBERT embedding dimension
        
        # Load LLM
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
    def index_dataset(self):
        dataset = load_dataset(self.config.dataset_name)
        
        for batch in dataset["train"].iter(batch_size=32):
            texts = [f"{doc['text']} {doc['caption']}" for doc in batch]
            embeddings = self.embeddings.encode(texts)
            self.index.add(embeddings, batch)
            
    def generate_response(self, query: str, k: int = 5):
        # Get query embedding and retrieve relevant documents
        query_embedding = self.embeddings.encode(query)
        relevant_docs = self.index.search(query_embedding, k=k)
        
        # Construct prompt with retrieved context
        context = "\n".join([doc["text"] for doc in relevant_docs])
        prompt = f"""Context: {context}\n\nQuestion: {query}\n\nAnswer:"""
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_length=512,
            num_return_sequences=1,
            temperature=0.7
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            "response": response,
            "relevant_docs": relevant_docs
        } 