# Fitness Question Answering System

This project implements a hybrid Question Answering (QA) system designed to return accurate and science-backed answers to fitness-related questions. It combines traditional Information Retrieval (IR) methods, TF-IDF and Word2Vec, with an extractive transformer model (fine-tuned DistilBERT) to retrieve and extract answers from a domain specific knowledge base.

## 🔍 Project Overview

- **Information Retrieval (IR):** Retrieves relevant documents using cosine similarity from either TF-IDF vectors or Word2Vec embeddings.
- **Extractive QA:** Uses a fine-tuned BERT-based model (DistilBERT) to extract answer spans from the top retrieved context.
- **Datasets Used:**
  - SQuAD 2.0 (benchmark dataset)
  - Fitness StackExchange (domain-specific dataset)

## 🧰 Technologies & Libraries

- Python 3.x
- `scikit-learn`
- `nltk`, `re`, `numpy`, `pandas`
- `gensim` (for Word2Vec)
- Hugging Face: `transformers`, `datasets`, `evaluate`
- `tensorflow` (for BERT fine-tuning)

## 📁 Project Structure

