# Fitness Question Answering System

This project implements a hybrid Question Answering (QA) system designed to return accurate and science-backed answers to fitness-related questions. It combines traditional Information Retrieval (IR) methods, TF-IDF and Word2Vec, with an extractive transformer model (fine-tuned DistilBERT) to retrieve and extract answers from a domain specific knowledge base.

## 🔍 Project Overview

- **Information Retrieval (IR):** Retrieves relevant documents using cosine similarity from either TF-IDF vectors or Word2Vec embeddings.
- **Extractive QA:** Uses a fine-tuned BERT-based model (DistilBERT) to extract answer spans from the top retrieved context.
- **Datasets Used:**
  - SQuAD 2.0 (benchmark dataset)
  - Fitness StackExchange (domain-specific dataset)
- All major components are implemented in **Jupyter Notebooks** for ease of experimentation and visualization.

## 🧰 Technologies & Libraries

- Python 3.x
- `scikit-learn`
- `nltk`, `re`, `numpy`, `pandas`
- `gensim` (for Word2Vec)
- Hugging Face: `transformers`, `datasets`, `evaluate`
- `tensorflow` (for BERT fine-tuning)

## 📁 Project Structure

```
project-root/
│
├── data/
│   ├── fitness_squad_filtered.json         # Final domain-specific QA dataset
│   ├── train.csv                           # SQuAD 2.0 training data
│   ├── validation.csv                      # SQuAD 2.0 validation data
│   ├── train.json                          # SQuAD 2.0 converted to JSON
│   └── validation.json                     # SQuAD 2.0 validation in JSON format
│
├── notebooks/
│   ├── 00_Parse_Fitness_XML.ipynb
│   ├── 00_Convert_SQuAD_CSV_to_JSON.ipynb
│   ├── 01_TFIDF_Word2Vec_QA.ipynb
│   ├── 02_BERT_Finetuning_SQuAD.ipynb
│
├── models/
│   └── word2vec_fitness.model              # Trained Word2Vec model
│
├── requirements.txt
└── README.md
```

## ▶️ How to Run

### 1. Set Up Environment

It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

### 2. Open the Notebooks

Run the notebooks in the following order:

1. **`02_BERT_Finetuning_SQuAD.ipynb`**  
   Fine-tunes the `distilbert-base-uncased` model on the SQuAD 2.0 dataset. Saves the model for later use.

2. **`01_TFIDF_Word2Vec_QA.ipynb`**  
   Contains:
   - TF-IDF and Word2Vec retrieval pipelines
   - Integration with the fine-tuned DistilBERT model
   - End-to-end QA pipeline using both general and domain-specific data

*Note: If training is already complete, you can skip notebook #2 and load the saved model directly. The model I fine-tuned has been published through the Hugging Face hub available in the notebook code or at https://huggingface.co/ekfrench/distilbert-finetuned-squad/tree/main*

### 3. Output Format

User input and output is interactive in the notebook. Example:

```bash
What is your fitness related question? 
Question input: How do I get a 6 pack abs?

Your Top Answers:
- single digit body fat and working out
- mixing cardio with strength training
```

## 📊 Evaluation

- **Retrieval:** Cosine similarity (TF-IDF, Word2Vec)
- **QA Model:** Exact Match and F1 Score (on SQuAD validation set)
- **Domain Evaluation:** Manual qualitative review on Fitness QA pairs

## ⚙️ Notes

- `top-k` context retrieval is configurable (default `k=2`)
- Fitness StackExchange data was filtered to only include answers with a score ≥ 2
- Preprocessing handled tokenization, cleaning, and XML-to-JSON conversion
- Due to file size constraints, raw data files (e.g., fitness.stackexchange.com.xml) are not included. To generate the required datasets:
  - Run 00_Parse_Fitness_XML.ipynb to create fitness_squad_filtered.json
  - Run 00_Convert_SQuAD_CSV_to_JSON.ipynb if you're working from CSV versions of SQuAD 2.0

## 📌 Licensing & Data Sources

- **SQuAD 2.0**: © Stanford NLP, used under public research license
- **Fitness StackExchange**: Public data dump from Internet Archive, used under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

## 🧠 Future Work

- Curate a high quality fitness-specific dataset 
- Integrate a generative QA model (e.g., GPT-3.5 or LLaMA)
- Deploy as a chatbot or API

---