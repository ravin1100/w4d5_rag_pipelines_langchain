import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class ArticleClassifier:
    """Article classifier using logistic regression."""
    
    def __init__(self, embedder):
        """
        Initialize classifier with specified embedder.
        
        Args:
            embedder: An instance of BaseEmbedder
        """
        self.embedder = embedder
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            multi_class='multinomial',
            max_iter=1000,
            random_state=42
        )
    
    def train(self, texts, labels):
        """
        Train the classifier.
        
        Args:
            texts (List[str]): Training texts
            labels (List[str]): Training labels
        """
        # Get embeddings
        X = self.embedder.batch_embed(texts)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, labels)
    
    def predict(self, text):
        """
        Predict category for a single text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Predicted category
            dict: Confidence scores for each category
        """
        # Get embedding
        X = self.embedder.embed(text).reshape(1, -1)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get prediction and probabilities
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        # Create confidence scores dictionary
        confidence_scores = {
            category: float(prob)
            for category, prob in zip(self.model.classes_, probabilities)
        }
        
        return prediction, confidence_scores
    
    def evaluate(self, texts, true_labels):
        """
        Evaluate the classifier.
        
        Args:
            texts (List[str]): Test texts
            true_labels (List[str]): True labels
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        # Get embeddings
        X = self.embedder.batch_embed(texts)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get predictions
        predictions = self.model.predict(X_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels,
            predictions,
            average='weighted'
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        } 