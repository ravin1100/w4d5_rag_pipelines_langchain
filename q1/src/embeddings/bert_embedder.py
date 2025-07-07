import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from .base_embedder import BaseEmbedder
from ..utils.text_preprocessing import preprocess_text

class BertEmbedder(BaseEmbedder):
    """BERT embedding model using the [CLS] token embeddings."""
    
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.dimension = self.model.config.hidden_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        print(f"Loaded {model_name} with dimension {self.dimension}")
    
    def embed(self, text):
        """
        Create document embedding using BERT [CLS] token.
        
        Args:
            text (str): Input text to embed
            
        Returns:
            np.ndarray: Document embedding vector
        """
        # Preprocess text
        processed_text = preprocess_text(text)
        
        # Tokenize and prepare input
        inputs = self.tokenizer(processed_text,
                              max_length=512,
                              truncation=True,
                              padding=True,
                              return_tensors='pt')
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding from the last hidden state
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embeddings[0]
    
    def batch_embed(self, texts):
        """
        Embed multiple texts.
        
        Args:
            texts (List[str]): List of input texts
            
        Returns:
            np.ndarray: Array of document embeddings
        """
        # Process texts in batches to avoid memory issues
        batch_size = 8
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            processed_texts = [preprocess_text(text) for text in batch_texts]
            
            # Tokenize and prepare input
            inputs = self.tokenizer(processed_texts,
                                  max_length=512,
                                  truncation=True,
                                  padding=True,
                                  return_tensors='pt')
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings) 