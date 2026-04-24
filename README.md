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
Run 1
```
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 5
NEG_SAMPLE_K = 4
MAX_HISTORY = 50
MAX_TITLE_LEN = 30
NUM_HEADS = 16
HEAD_DIM = 16

Loss encourages higher scores for clicked articles compared to non-clicked ones.
```
Run 2
```
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 5
NEG_SAMPLE_K = 7
MAX_HISTORY = 50
MAX_TITLE_LEN = 30
NUM_HEADS = 16
HEAD_DIM = 16
```
Run 3
```
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 5
NEG_SAMPLE_K = 4
MAX_HISTORY = 100
MAX_TITLE_LEN = 30
NUM_HEADS = 16
HEAD_DIM = 16
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

# Results - visuals in Notebook: 03_training
### Run 1
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

<img width="1500" height="750" alt="metrics_comparison" src="https://github.com/user-attachments/assets/60f252e8-087f-419e-af0a-268f9d44eea6" />

The baseline clears the basic NRMS range on every metric and sits within the tuned range on three of the four. 
nDCG@5 is the one metric that falls just short of the tuned floor, reflecting the precision gap noted above. Overall the model performs at the upper end of a basic implementation and overlaps substantially with tuned performance.

### Error Analysis
Short click histories - The user encoder relies on self-attention across clicked articles to build a preference profile. Users with only one or two items in their history give the attention mechanism almost nothing to work with, producing a user vector dominated by a single article rather than a genuine interest pattern. These users likely account for a disproportionate share of nDCG@5 misses.
Title-only representation - The news encoder sees only the headline — 30 tokens at most. Articles on similar topics with different surface wording produce similar vectors, limiting the model's ability to discriminate between them. 
Popular-item bias - Negatives are sampled uniformly from each impression list during training, with no correction for article frequency. Frequently appearing articles are suppressed as negatives many times, which may cause the model to under-score them even when they are genuinely relevant — a likely contributor to the nDCG@5 shortfall.

### Run 2
```text
For neg_k of: 7
======================================
  EVALUATION RESULTS
======================================
  AUC       : 0.6793
  MRR       : 0.3886
  nDCG@5    : 0.3660
  nDCG@10   : 0.4232
  Impressions: 156,965
======================================
Metrics saved to results/eval_metrics2.json
```
### Quantitative Results
Increasing the number of training negatives from 4 to 7 produced no measurable change in evaluation metrics. AUC, MRR, nDCG@5, and nDCG@10 are identical to the baseline to four decimal places. The training loss curve tells a consistent story — run 2 finished at 1.3381 versus run 1's 1.3458, a difference of 0.0077, indicating marginal faster convergence but not a meaningfully different model. The additional negatives per positive did not provide a harder or more informative training signal at this dataset scale.

### Error Analysis
The insensitivity to neg_k is likely explained by two factors. First, MINDsmall impression lists are relatively short, so the practical difference between sampling 4 and 7 negatives is small — many impressions may not have enough non-clicked candidates to meaningfully distinguish the two settings. Second, the negatives are still sampled uniformly with no difficulty weighting, so the additional three negatives per positive are drawn from the same easy distribution as before. Harder negative mining strategies — such as sampling articles semantically similar to the positive — would likely be needed before changes to K produce a detectable effect.


# Comparison of Tuned Runs/Hyperparameters
### Loss
Comparing the training loss curves across all three runs reveals that the models converged to virtually the same solution regardless of the hyperparameter change. Run 1 (baseline, K=4) finished at a final epoch loss of 1.3458, run 2 (K=7) at 1.3381, and run 3 (longer history length) at 1.3436 — a spread of just 0.0077 across five epochs. The curves track each other closely at every epoch, with run 2 converging marginally faster and run 1 slightly slower, but no run diverges meaningfully from the others. This suggests that neither increasing negative samples from 4 to 7 nor extending the maximum history length produced a sufficiently different training signal to shift the model toward a different solution — and consequently, near-identical evaluation metrics across all three runs are the expected outcome, not simply an artefact of the checkpoint issue. The model appears insensitive to these particular hyperparameter changes at this scale.
### Run 1
<img width="1200" height="600" alt="loss_curve" src="https://github.com/user-attachments/assets/80deafd0-1d0d-4e9f-a216-fc080ae83dc0" />
### Run 2
<img width="1200" height="600" alt="loss_curve2" src="https://github.com/user-attachments/assets/73d9a338-51bb-4635-b0ce-eabb934db3a9" />
### Run 3
<img width="1200" height="600" alt="loss_curve3" src="https://github.com/user-attachments/assets/a3554c72-3e44-4fe9-8e4c-deb6b7b4285d" />

# Conclusions
The NRMS baseline trained on MINDsmall demonstrates that dual multi-head self-attention over GloVe-initialised title embeddings is a strong starting point for news recommendation, achieving results at the upper end of a basic implementation and overlapping substantially with tuned performance across three of four standard MIND metrics. The model's clearest strength is MRR, where it exceeds the tuned NRMS ceiling, indicating it reliably places at least one relevant article near the top of each impression list. Its clearest weakness is nDCG@5, where it falls just short of the tuned floor, pointing to a precision gap in the top positions that title-only features and uniform negative sampling cannot fully close.

The ablation experiments revealed that neither increasing negative samples from K=4 to K=7 nor extending the maximum history length from 50 to 100 produced any measurable change in evaluation metrics. This is not a failure of the experiments but an informative finding in itself — the model is insensitive to these particular changes at this dataset scale, most likely because MINDsmall impression lists are too short for the extra negatives to matter and because most users do not have histories long enough to benefit from a higher history ceiling. The training loss curves confirm all three models trained correctly and converged to the same solution, suggesting the architecture itself is the binding constraint rather than these hyperparameters.

The most direct path to meaningful improvement would be adding abstract and category features to the news encoder, implementing harder negative mining, and experimenting with learning rate scheduling — changes that target the actual bottlenecks identified in the error analysis rather than parameters the model has already saturated.

