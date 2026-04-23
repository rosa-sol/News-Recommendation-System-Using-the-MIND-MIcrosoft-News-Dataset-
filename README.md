# News Recommendation-System Using the MIND MIcrosoft News Dataset
News Recommendation System using MIND Dataset

Course: Machine Learning Capstone
Instructor: Sohaib Kiani
Estimated Duration: 3–4 Weeks

## 1. Introduction/Project Overview

This project develops an end-to-end news recommendation system using the MIND (Microsoft News Dataset). The objective is to predict user clicks on news articles based on historical user behavior and article content.

News recommendation presents unique challenges due to rapidly changing user interests, short-lived content relevance, and sparsity of interaction data. This project implements a neural recommendation pipeline inspired by NRMS (Neural News Recommendation with Multi-Head Self-Attention).

The workflow includes data preprocessing, feature engineering, model development, training, evaluation, and analysis.

## Learning Objectives
```text
By completing this project, the following skills are developed:

Understanding recommendation system pipelines
Working with large-scale behavioral datasets
Applying NLP techniques to news content
Implementing deep learning-based recommender systems in PyTorch
Evaluating ranking models using standard metrics
Performing error analysis and model interpretation
```
## Dataset Description (MIND)

The MIND dataset is a large-scale news recommendation dataset released by Microsoft Research. It contains anonymized user interaction logs collected from Microsoft News.

Dataset Versions
Version	Users	Articles	Impressions	Clicks
MIND-small	50,000	~65,000	~230,000	~347,000
MIND-large	1,000,000	~160,000	~15,000,000	~24,000,000

This project uses MIND-small for computational feasibility.

# Key Files

behaviors.tsv

User impression logs
Fields: user ID, timestamp, click history, impressions, news.tsv

# 2. Methodology:
## Project Structure

```text
mind-recommender/
├── data/
│   ├── MINDsmall_train/      # Training set (behaviors, news, entities)
│   ├── MINDsmall_dev/        # Validation set
│   └── glove/                # Pre-trained word embeddings
├── src/
│   ├── data_loader.py        # Custom Dataset and DataLoader classes
│   ├── news_encoder.py       # PLM or CNN-based news representation
│   ├── user_encoder.py       # GRU/Attention-based user history encoding
│   ├── model.py              # Recommender architecture (e.g., NRMS, LSTUR)
│   ├── train.py              # Model training loop and optimization
│   └── evaluate.py           # Metrics (AUC, MRR, nDCG@k)
├── notebooks/
│   ├── 01_eda.ipynb          # Dataset statistics and word clouds
│   ├── 02_preprocessing.ipynb # Tokenization and embedding mapping
│   └── 03_training.ipynb     # Interactive training and visualization
├── models/                   # Saved .pth or .h5 model weights
├── results/                  # Generated plots and performance logs
└── README.md                 # Project overview and setup instructions
```
## Exploratory Data Analysis

The dataset was analyzed to understand user behavior and content structure.

### Key analyses:

Click-through rate (CTR)
Category and subcategory distributions
User activity distribution
Impression size analysis
Temporal click patterns (hourly and daily)
Text statistics (title and abstract lengths)

### Findings:

Strong morning engagement peak around 08:00
Highly skewed user activity distribution
Clear imbalance across news categories
Event-driven spikes in daily click behavior
4.3 Data Preprocessing and Feature Engineering

### Steps performed:
```text
Tokenization of news titles
Vocabulary construction with frequency filtering
Mapping tokens to GloVe embeddings
Fixed-length padding and truncation of titles
Parsing user behavior sequences
Generation of training samples using negative sampling
Train/validation split following temporal order
```
### Each training sample consists of:

User click history
One positive news article
Multiple negative sampled articles
4.4 Model Architecture

## The model is based on an NRMS-style neural recommendation architecture.
```text
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
```
### Key settings:
```
Optimizer: Adam
Learning rate: 1e-4
Batch size: 64
Negative samples per positive: 4
Epochs: 5
Gradient clipping applied for stability

Loss encourages higher scores for clicked articles compared to non-clicked ones.
```
### Evaluation and Metrics
```
The model is evaluated using standard recommendation metrics:

AUC (Area Under ROC Curve)
MRR (Mean Reciprocal Rank)
nDCG@5
nDCG@10
```
These metrics evaluate ranking quality and the model’s ability to prioritize relevant news articles.

Note: Still working on assignment had to re-run my training loop. Will be done by 4/22.

# Results 
```text
For learning rate of: 1×10⁻⁴
======================================
  EVALUATION RESULTS
======================================
  AUC       : 0.6793
  MRR       : 0.3886
  nDCG@5    : 0.3660
  nDCG@10   : 0.4232
  Impressions: 156,965
======================================
Metrics saved to results/eval_metrics.json
```
### Quantitative Results
The baseline model was evaluated across 156,965 dev-set impressions using four standard MIND benchmark metrics.
AUC (0.6793) measures how often the model ranks a clicked article above a non-clicked one — here, about 68% of the time. The 18-point margin above random (0.50) confirms the model has learned genuine user preference from title text alone, though roughly one in three pairwise comparisons still goes the wrong way.
MRR (0.3886) places the first relevant article at roughly rank 2–3 on average. The model frequently promotes the right article into the top handful but does not reliably push it to position 1 — meaningful for interfaces that surface a ranked list, more problematic for those showing only a single result.
nDCG@5 (0.3660) and nDCG@10 (0.4232) are most informative together. The gap of ~0.057 between the two cutoffs indicates that a meaningful share of relevant articles land in positions 6–10 rather than the top 5. The model is finding the right content but not surfacing it aggressively enough — it ranks correctly in a broad sense but lacks the precision to concentrate hits at the very top.

### Comparison with Established Baselines
Comparison with Established Baselines
| Metric | Random | Basic NRMS | Tuned NRMS | Run 1 | 
|------|-------|-----|-------|----|
| AUC | ~0.500 | 0.62–0.66 | 0.67–0.70 | 0.6793 |
| MMR | ~0.200 | 0.28–0.31 | 0.31–0.34 | 0.3886 |
| nDCG@5 | ~0.200 | 0.30–0.34 | 0.34–0.38 | 0.3660 |
| nDCG@10 | ~0.300 | 0.36–0.40 | 0.40–0.44 | 0.4232 |

The baseline clears the basic NRMS range on every metric and sits within the tuned range on three of the four. 
nDCG@5 is the one metric that falls just short of the tuned floor, reflecting the precision gap noted above. Overall the model performs at the upper end of a basic implementation and overlaps substantially with tuned performance.

### Error Analysis
Short click histories. The user encoder relies on self-attention across clicked articles to build a preference profile. Users with only one or two items in their history give the attention mechanism almost nothing to work with, producing a user vector dominated by a single article rather than a genuine interest pattern. These users likely account for a disproportionate share of nDCG@5 misses.
Title-only representation. The news encoder sees only the headline — 30 tokens at most. Articles on similar topics with different surface wording produce similar vectors, limiting the model's ability to discriminate between them. Abstract and category features, omitted here, carry substantial disambiguating signal and are the most direct explanation for the remaining gap to tuned NRMS.
Popular-item bias. Negatives are sampled uniformly from each impression list during training, with no correction for article frequency. Frequently appearing articles are suppressed as negatives many times, which may cause the model to under-score them even when they are genuinely relevant — a likely contributor to the nDCG@5 shortfall.

What the metrics do not capture. AUC, MRR, and nDCG are all rank-based and treat every impression independently. They do not measure diversity, novelty, or whether the model recommends the same few articles repeatedly across users — practical failure modes invisible in these numbers.

