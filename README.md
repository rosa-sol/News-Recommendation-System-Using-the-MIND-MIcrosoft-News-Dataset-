# News-Recommendation-System-Using-the-MIND-MIcrosoft-News-Dataset-
News Recommendation System using MIND Dataset

Course: Machine Learning Capstone
Instructor: Sohaib Kiani
Estimated Duration: 3–4 Weeks

## 1. Project Overview

This project develops an end-to-end news recommendation system using the MIND (Microsoft News Dataset). The objective is to predict user clicks on news articles based on historical user behavior and article content.

News recommendation presents unique challenges due to rapidly changing user interests, short-lived content relevance, and sparsity of interaction data. This project implements a neural recommendation pipeline inspired by NRMS (Neural News Recommendation with Multi-Head Self-Attention).

The workflow includes data preprocessing, feature engineering, model development, training, evaluation, and analysis.

## 2. Learning Objectives

By completing this project, the following skills are developed:

Understanding recommendation system pipelines
Working with large-scale behavioral datasets
Applying NLP techniques to news content
Implementing deep learning-based recommender systems in PyTorch
Evaluating ranking models using standard metrics
Performing error analysis and model interpretation

## 3. Dataset Description (MIND)

The MIND dataset is a large-scale news recommendation dataset released by Microsoft Research. It contains anonymized user interaction logs collected from Microsoft News.

Dataset Versions
Version	Users	Articles	Impressions	Clicks
MIND-small	50,000	~65,000	~230,000	~347,000
MIND-large	1,000,000	~160,000	~15,000,000	~24,000,000

This project uses MIND-small for computational feasibility.

# Key Files

behaviors.tsv

User impression logs
Fields: user ID, timestamp, click history, impressions

news.tsv

## Project structure:
"""
mind-recommender/
├── data/
│   ├── MINDsmall_train/
│   ├── MINDsmall_dev/
│   └── glove/
├── src/
│   ├── data_loader.py
│   ├── news_encoder.py
│   ├── user_encoder.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_training.ipynb
├── models/
├── results/
└── README.md
"""
## 4.2 Exploratory Data Analysis

The dataset was analyzed to understand user behavior and content structure.

Key analyses:

Click-through rate (CTR)
Category and subcategory distributions
User activity distribution
Impression size analysis
Temporal click patterns (hourly and daily)
Text statistics (title and abstract lengths)

Findings:

Strong morning engagement peak around 08:00
Highly skewed user activity distribution
Clear imbalance across news categories
Event-driven spikes in daily click behavior
4.3 Data Preprocessing and Feature Engineering

Steps performed:

Tokenization of news titles
Vocabulary construction with frequency filtering
Mapping tokens to GloVe embeddings
Fixed-length padding and truncation of titles
Parsing user behavior sequences
Generation of training samples using negative sampling
Train/validation split following temporal order

Each training sample consists of:

User click history
One positive news article
Multiple negative sampled articles
4.4 Model Architecture

The model is based on an NRMS-style neural recommendation architecture.

News Encoder
Word embedding layer (GloVe)
Multi-head self-attention
Additive attention pooling
User Encoder
Encodes sequence of clicked news articles
Shared news encoder weights
Attention-based aggregation of user history
Scoring Function
Dot product between user representation and candidate news vectors

Output:

Relevance scores for candidate articles
4.5 Training Procedure

Training is performed using cross-entropy loss over candidate impressions.

Key settings:

Optimizer: Adam
Learning rate: 1e-4
Batch size: 64
Negative samples per positive: 4
Epochs: 5
Gradient clipping applied for stability

Loss encourages higher scores for clicked articles compared to non-clicked ones.

4.6 Evaluation and Metrics

The model is evaluated using standard recommendation metrics:

AUC (Area Under ROC Curve)
MRR (Mean Reciprocal Rank)
nDCG@5
nDCG@10

These metrics evaluate ranking quality and the model’s ability to prioritize relevant news articles.

5. Project Structure
mind-recommender/
├── data/
│   ├── MINDsmall_train/
│   ├── MINDsmall_dev/
│   └── glove/
├── src/
│   ├── data_loader.py
│   ├── news_encoder.py
│   ├── user_encoder.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_training.ipynb
├── models/
├── results/
└── README.md
