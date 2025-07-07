import numpy as np
from sentence_transformers import SentenceTransformer
from .base_embedder import BaseEmbedder
from ..utils.text_preprocessing import preprocess_text

class SentenceBertEmbedder(BaseEmbedder):
    """Sentence-BERT embedding model."""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        super().__init__()
        print(f"Loading {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Loaded {model_name} with dimension {self.dimension}")
    
    def embed(self, text):
        """
        Create document embedding using Sentence-BERT.
        
        Args:
            text (str): Input text to embed
            
        Returns:
            np.ndarray: Document embedding vector
        """
        # Preprocess text
        processed_text = preprocess_text(text)
        
        # Get embedding
        embedding = self.model.encode(processed_text, convert_to_numpy=True)
        return embedding
    
    def batch_embed(self, texts):
        """
        Embed multiple texts.
        
        Args:
            texts (List[str]): List of input texts
            
        Returns:
            np.ndarray: Array of document embeddings
        """
        # Preprocess texts
        processed_texts = [preprocess_text(text) for text in texts]
        
        # Get embeddings
        embeddings = self.model.encode(processed_texts, convert_to_numpy=True)
        return embeddings 