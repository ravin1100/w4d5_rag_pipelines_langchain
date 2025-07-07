import os
import numpy as np
import openai
from dotenv import load_dotenv
from .base_embedder import BaseEmbedder
from ..utils.text_preprocessing import preprocess_text

# Load environment variables
load_dotenv()

class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embedding model using text-embedding-ada-002."""
    
    def __init__(self):
        super().__init__()
        self.model = "text-embedding-ada-002"
        self.dimension = 1536  # Fixed dimension for ada-002
        
        # Set up OpenAI API key
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in .env file")
        
        print(f"Initialized OpenAI embedder with model {self.model}")
    
    def embed(self, text):
        """
        Create document embedding using OpenAI's API.
        
        Args:
            text (str): Input text to embed
            
        Returns:
            np.ndarray: Document embedding vector
        """
        # Preprocess text
        processed_text = preprocess_text(text)
        
        # Get embedding from API
        response = openai.Embedding.create(
            input=processed_text,
            model=self.model
        )
        
        # Extract embedding
        embedding = response['data'][0]['embedding']
        return np.array(embedding)
    
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
        
        # Get embeddings from API
        response = openai.Embedding.create(
            input=processed_texts,
            model=self.model
        )
        
        # Extract embeddings
        embeddings = [data['embedding'] for data in response['data']]
        return np.array(embeddings) 