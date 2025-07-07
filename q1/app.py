import streamlit as st
import plotly.express as px
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import pandas as pd
from sklearn.preprocessing import StandardScaler
import json

from src.embeddings.word2vec_embedder import Word2VecEmbedder
from src.embeddings.bert_embedder import BertEmbedder
from src.embeddings.sentence_bert_embedder import SentenceBertEmbedder
from src.embeddings.openai_embedder import OpenAIEmbedder
from src.models.classifier import ArticleClassifier
from src.utils.text_preprocessing import get_categories

def load_training_data():
    """Load training data from JSON file."""
    try:
        with open('data/training_data.json', 'r') as f:
            data = json.load(f)
            texts = [item['text'] for item in data['data']]
            labels = [item['category'] for item in data['data']]
            return texts, labels
    except FileNotFoundError:
        st.error("Training data file not found. Using sample data instead.")
        return get_sample_data()

def get_sample_data():
    """Get minimal sample data for testing."""
    sample_texts = [
        "Apple announces new iPhone with revolutionary AI capabilities",
        "Stock market reaches record high as tech sector booms",
        "New cancer treatment shows promising results in clinical trials",
        "Champions League final ends in dramatic penalty shootout",
        "Senate passes landmark climate change legislation",
        "Hollywood stars gather for Academy Awards ceremony"
    ]
    sample_labels = [
        "Tech", "Finance", "Healthcare", "Sports", "Politics", "Entertainment"
    ]
    return sample_texts, sample_labels

# Page config
st.set_page_config(page_title="Smart Article Categorizer", layout="wide")

# Title
st.title("Smart Article Categorizer")
st.markdown("""
This app uses multiple embedding models to classify articles into categories:
- Tech
- Finance
- Healthcare
- Sports
- Politics
- Entertainment
""")

# Initialize session state
if 'embedders' not in st.session_state:
    with st.spinner("Loading embedding models..."):
        st.session_state.embedders = {
            'Word2Vec': Word2VecEmbedder(),
            'BERT': BertEmbedder(),
            'Sentence-BERT': SentenceBertEmbedder(),
            'OpenAI': OpenAIEmbedder()
        }
        st.session_state.classifiers = {
            name: ArticleClassifier(embedder)
            for name, embedder in st.session_state.embedders.items()
        }
        st.session_state.is_trained = False

# Training section
st.header("Model Training")

if not st.session_state.is_trained:
    st.warning("⚠️ Models are not trained yet. Please use the training data to train the models.")
    
    # Training data options
    if st.button("Train Models"):
        with st.spinner("Training models with available data..."):
            # Load training data
            train_texts, train_labels = load_training_data()
            
            # Train all classifiers
            for classifier in st.session_state.classifiers.values():
                classifier.train(train_texts, train_labels)
            
            st.session_state.is_trained = True
            st.success("✅ Models trained successfully!")
            st.experimental_rerun()
else:
    st.success("✅ Models are trained and ready to use!")

# Input section
st.header("Article Classification")

# Input text
text_input = st.text_area(
    "Enter article text:",
    height=200,
    help="Paste your article text here for classification"
)

# Classify button
if st.button("Classify") and text_input:
    if not st.session_state.is_trained:
        st.error("⚠️ Please train the models first!")
    else:
        # Create columns for results
        cols = st.columns(len(st.session_state.embedders))
        
        # Get predictions from each model
        embeddings_dict = {}  # Store embeddings with their dimensions
        categories = get_categories()
        
        for (name, classifier), col in zip(st.session_state.classifiers.items(), cols):
            with col:
                st.subheader(f"{name} Model")
                
                # Get prediction and confidence scores
                prediction, confidence_scores = classifier.predict(text_input)
                
                # Display prediction
                st.markdown(f"**Prediction:** {prediction}")
                
                # Create confidence score chart
                df = pd.DataFrame({
                    'Category': list(confidence_scores.keys()),
                    'Confidence': list(confidence_scores.values())
                })
                
                fig = px.bar(
                    df,
                    x='Category',
                    y='Confidence',
                    title='Confidence Scores',
                    range_y=[0, 1]
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Store embedding
                embedding = classifier.embedder.embed(text_input)
                embeddings_dict[name] = embedding
        
        # Visualization of embeddings
        st.subheader("Embedding Visualizations")
        
        try:
            # First, reduce each embedding to 50 dimensions using PCA if larger
            processed_embeddings = []
            for name, embedding in embeddings_dict.items():
                # Ensure 2D array shape
                if embedding.ndim == 1:
                    embedding = embedding.reshape(1, -1)
                
                # If embedding dimension is larger than 50, reduce it
                if embedding.shape[1] > 50:
                    pca = PCA(n_components=50)
                    embedding = pca.fit_transform(embedding)
                
                # Normalize
                scaler = StandardScaler()
                normalized = scaler.fit_transform(embedding)
                processed_embeddings.append(normalized.flatten())
            
            # Convert to numpy array
            processed_embeddings = np.stack(processed_embeddings)
            
            # Create tabs for different visualization methods
            viz_tabs = st.tabs(["t-SNE", "UMAP"])
            
            with viz_tabs[0]:  # t-SNE
                # Apply t-SNE
                tsne = TSNE(n_components=2, random_state=42, perplexity=2)  # Lower perplexity for small dataset
                embeddings_2d = tsne.fit_transform(processed_embeddings)
                
                # Create scatter plot
                df = pd.DataFrame({
                    'x': embeddings_2d[:, 0],
                    'y': embeddings_2d[:, 1],
                    'Model': list(embeddings_dict.keys())
                })
                
                fig = px.scatter(
                    df,
                    x='x',
                    y='y',
                    color='Model',
                    title='t-SNE Visualization of Embeddings'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with viz_tabs[1]:  # UMAP
                # Apply UMAP
                reducer = umap.UMAP(random_state=42, n_neighbors=2)  # Lower n_neighbors for small dataset
                embeddings_2d = reducer.fit_transform(processed_embeddings)
                
                # Create scatter plot
                df = pd.DataFrame({
                    'x': embeddings_2d[:, 0],
                    'y': embeddings_2d[:, 1],
                    'Model': list(embeddings_dict.keys())
                })
                
                fig = px.scatter(
                    df,
                    x='x',
                    y='y',
                    color='Model',
                    title='UMAP Visualization of Embeddings'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error in visualization: {str(e)}")
            st.error("Could not generate embedding visualizations, but classifications are still valid.")

else:
    st.info("Enter some text and click 'Classify' to see the predictions from different models.") 