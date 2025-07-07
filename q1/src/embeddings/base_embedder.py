from abc import ABC, abstractmethod
import numpy as np

class BaseEmbedder(ABC):
    """Base class for all embedding models."""
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.dimension = None
    
    @abstractmethod
    def embed(self, text):
        """
        Convert text into embeddings.
        
        Args:
            text (str): Input text to embed
            
        Returns:
            np.ndarray: Embedding vector
        """
        pass
    
    @abstractmethod
    def batch_embed(self, texts):
        """
        Convert multiple texts into embeddings.
        
        Args:
            texts (List[str]): List of input texts to embed
            
        Returns:
            np.ndarray: Array of embedding vectors
        """
        pass
    
    def get_dimension(self):
        """Return the dimension of the embeddings."""
        return self.dimension 