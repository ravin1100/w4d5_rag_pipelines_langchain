import numpy as np
from gensim.models import KeyedVectors
from gensim.downloader import load
from .base_embedder import BaseEmbedder
from ..utils.text_preprocessing import preprocess_text, split_into_sentences

class Word2VecEmbedder(BaseEmbedder):
    """Word2Vec embedding model using pre-trained vectors."""
    
    def __init__(self, model_name='word2vec-google-news-300'):
        super().__init__()
        print(f"Loading {model_name}...")
        self.model = load(model_name)
        self.dimension = self.model.vector_size
        print(f"Loaded {model_name} with dimension {self.dimension}")
    
    def embed(self, text):
        """
        Create document embedding by averaging word vectors.
        
        Args:
            text (str): Input text to embed
            
        Returns:
            np.ndarray: Document embedding vector
        """
        # Preprocess text
        processed_text = preprocess_text(text)
        words = processed_text.split()
        
        # Get word vectors and average them
        word_vectors = []
        for word in words:
            try:
                vector = self.model[word]
                word_vectors.append(vector)
            except KeyError:
                continue
        
        if not word_vectors:
            return np.zeros(self.dimension)
        
        return np.mean(word_vectors, axis=0)
    
    def batch_embed(self, texts):
        """
        Embed multiple texts.
        
        Args:
            texts (List[str]): List of input texts
            
        Returns:
            np.ndarray: Array of document embeddings
        """
        return np.array([self.embed(text) for text in texts]) 