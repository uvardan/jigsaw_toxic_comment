# Capstone Project: Toxic Comment Triage with EDA and Baseline Modeling

## Project Overview
This capstone explores a human-in-the-loop text triage workflow using the Jigsaw Toxic Comment Classification dataset. The goal is to route comments into one of three operational classes:

- **safe**
- **needs_review**
- **high_risk**

This framing supports practical moderation by separating clearly safe content from severe risk and ambiguous cases that may need human review.

## Research Question
Can a baseline machine learning model, supported by exploratory data analysis (EDA) and simple feature engineering, effectively triage toxic comments into **safe**, **needs_review**, and **high_risk** categories?

## Dataset
- **Source:** Kaggle - Jigsaw Toxic Comment Classification Challenge
- **Primary file used:** `train.csv`
- **Core columns:**
  - `comment_text`
  - `toxic`
  - `severe_toxic`
  - `obscene`
  - `threat`
  - `insult`
  - `identity_hate`

## Label Mapping
The original binary toxicity columns were transformed into a 3-class triage label:

- **safe**: all toxicity labels are 0
- **high_risk**: `threat = 1` or `severe_toxic = 1` or `identity_hate = 1`
- **needs_review**: all other toxic comments not classified as high-risk

## Methodology
### 1. Data Cleaning
- Verified required columns were present
- Checked missing values
- Checked for duplicate rows and duplicate `comment_text`
- Removed duplicate comments
- Applied light text cleaning:
  - lowercasing
  - URL replacement
  - email replacement
  - whitespace normalization

### 2. Feature Engineering
Created simple interpretable text features, including:
- character count
- word count
- exclamation count
- question count
- digit count
- uppercase count
- uppercase ratio

### 3. Exploratory Data Analysis (EDA)
Performed EDA to better understand the data and relationships between variables:
- class distribution of triage labels
- summary statistics of engineered features by class
- word count distribution by class
- character count distribution by class
- top unigrams by class
- top bigrams by class
- qualitative sample comments from each class

### 4. Baseline Model
Used a baseline text classification pipeline:
- **TF-IDF vectorization** (word unigrams and bigrams)
- **Logistic Regression** with `class_weight="balanced"`

### 5. Evaluation Metric
The primary evaluation metric is **macro F1-score**.

**Why macro F1?**
The triage classes are highly imbalanced, so macro F1 gives equal weight to each class instead of letting the majority `safe` class dominate the score.

## Results
### EDA Summary
- The derived triage labels are strongly imbalanced:
  - **safe:** 143,346 (89.83%)
  - **needs_review:** 13,238 (8.30%)
  - **high_risk:** 2,987 (1.87%)
- This imbalance supports using **macro F1** instead of accuracy alone.
- `needs_review` overlaps behaviorally with both `safe` and `high_risk`, making it the most ambiguous class.
- Engineered text features show differences across classes, especially around punctuation intensity and uppercase usage.
- N-gram analysis highlights stronger toxic lexical signals in `high_risk`, while `needs_review` includes more borderline language patterns.

### Baseline Model Performance
Using **TF-IDF + Logistic Regression** on a stratified train/test split:

- **Macro F1-score:** **0.6936**
- **Accuracy:** **0.9257**

Per-class results:
- **high_risk** F1: **0.4973**
- **needs_review** F1: **0.6159**
- **safe** F1: **0.9676**

### Interpretation
- The baseline performs very well on the dominant `safe` class.
- Performance is weaker on `high_risk`, which is expected because it is the rarest class.
- The biggest challenge is correctly handling ambiguous cases, which supports the project's human-in-the-loop triage framing.

## Key Takeaways
- A triage-based moderation workflow is feasible with a classical NLP baseline.
- The main modeling challenge is class imbalance and ambiguity between `needs_review` and `high_risk`.
- The current baseline is strong enough to act as a comparison point for Module 24.

## Next Steps (Module 24)
- Test additional models for comparison (for example, Linear SVM or Naive Bayes)
- Improve feature engineering and preprocessing
- Add more structured error analysis and threshold tuning
- Refine notebook presentation for technical and non-technical audiences

## Repository Structure
A clean recommended structure:

```text
capstone-project/
├── README.md
├── notebooks/
│   └── 01_jigsaw_eda_and_cleaning.ipynb
├── data/
│   └── train.csv
├── .gitignore
└── requirements.txt
```

## Jupyter Notebook
Main analysis notebook:
- `notebooks/01_jigsaw_eda_and_cleaning.ipynb`

After uploading to GitHub, replace the line below with your actual notebook link if desired:
- Notebook link: `https://github.com/<your-username>/<your-repo>/blob/main/notebooks/01_jigsaw_eda_and_cleaning.ipynb`

## How to Run
1. Clone the repository
2. Install dependencies
3. Place `train.csv` in the `data/` folder
4. Open and run the notebook in Jupyter

## Submission Note
For this module, the primary deliverables are:
- the Jupyter notebook(s)
- the updated `README.md`
- the GitHub repository link submitted to the course portal
