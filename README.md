# Capstone Project: Risk-Aware Toxic Comment Triage System

## Project Overview
This capstone builds a **risk-aware, human-in-the-loop text classification system** using the Jigsaw Toxic Comment dataset.  

The goal is not only to classify text, but to **support real-world decision-making** by routing content into three operational categories:

- **safe** → automatically processed  
- **needs_review** → sent to human reviewers  
- **high_risk** → flagged for immediate action  

This framing reflects practical moderation systems used in real-world AI applications.

---

## Research Question
How can a machine learning system classify user-generated text into safe, needs human review, and high-risk categories to support reliable and efficient human decision-making?

---

## Dataset
- **Source:** Kaggle – Jigsaw Toxic Comment Classification Challenge  
- **Size:** ~159k comments  
- **Labels:** multi-label toxicity indicators  

### Original Labels
- toxic  
- severe_toxic  
- obscene  
- threat  
- insult  
- identity_hate  

---

## Triage Label Mapping
The original multi-label dataset was transformed into a **3-class triage system**:

- **safe** → no toxicity labels  
- **high_risk** → threat OR severe_toxic OR identity_hate  
- **needs_review** → all remaining toxic content  

This mapping enables a **decision-focused classification system**, not just a prediction model.

---

## Methodology

### 1. Data Preparation
- Removed missing values and duplicates  
- Cleaned text (lowercase, URLs, emails, whitespace normalization)

---

### 2. Feature Engineering
- TF-IDF (unigrams + bigrams)
- Lightweight interpretable features (word count, punctuation, etc.)

---

### 3. Exploratory Data Analysis (EDA)
Key findings:
- Severe **class imbalance** (safe dominates)
- **needs_review** is the most ambiguous class
- Toxicity is often expressed via **phrases (n-grams)** rather than single words
- Text length is not a strong predictor

---

### 4. Baseline Model
- TF-IDF + Logistic Regression  
- Stratified train/test split  
- Evaluated using **macro F1-score**

---

### 5. Model Improvements

#### Class Weighting
- Applied `class_weight="balanced"`
- Improved recall for minority classes

#### Threshold Tuning
- Custom decision thresholds applied to probabilities
- Prioritized **high-risk detection over accuracy**

---

## Model Comparison

| Model | Accuracy | Macro F1 | High-Risk Recall |
|------|---------|----------|------------------|
| Baseline | ~0.94 | ~0.67 | 0.30 |
| Class Weighted | ~0.90 | ~0.66 | 0.62 |
| Threshold Tuned | ~0.88 | ~0.62 | 0.74 |

### Key Insight
The system was intentionally optimized for **recall on high-risk content**, even at the cost of precision and accuracy.

---

## Error Analysis

Key failure patterns:
- Over-reliance on keywords
- Difficulty with **sarcasm and tone**
- Misclassification of **context-dependent language**
- Confusion between **needs_review vs high_risk**

---

## Synthetic Edge Case Testing

Custom test cases were created to evaluate:
- sarcasm
- ambiguity
- borderline toxicity

Findings:
- Strong performance on clear cases
- Weak performance on sarcasm and tone-dependent language

---

## System Design

Pipeline:

1. Input text  
2. Text preprocessing  
3. TF-IDF feature extraction  
4. Logistic Regression prediction (probabilities)  
5. Threshold-based decision logic  
6. Output routing  

---

## Human-in-the-Loop Workflow

- **safe** → auto-approved  
- **needs_review** → sent to human reviewer  
- **high_risk** → flagged for immediate action  

This ensures:
- efficiency for safe content  
- safety for harmful content  
- oversight for ambiguous cases  

---

## Key Takeaways

- A simple, interpretable model can support real-world moderation workflows  
- **Class imbalance and ambiguity are the main challenges**  
- **Threshold tuning is critical** for risk-sensitive systems  
- Human review is essential for contextual understanding  

---

## Limitations

- Cannot understand **context, sarcasm, or tone**
- Relies heavily on **keywords**
- Struggles with **ambiguous cases**
- Linear model limits complex pattern learning

---

## Future Improvements

- Use transformer models (e.g., BERT) for contextual understanding  
- Improve dataset balance  
- Add reviewer feedback loop  
- Deploy as a real-time moderation API  

---

## Why This Project Matters

In real-world systems, AI does not replace humans—it **augments decision-making**.

This project demonstrates how machine learning can:
- reduce manual workload  
- prioritize risk  
- enable scalable moderation  

---

## Repository Structure
jigsaw_toxic_comment/
├── README.md
├── 01_jigsaw_eda_and_cleaning.ipynb
├── data/
│   └── train.csv


---

## How to Run

1. Clone repository  
2. Install dependencies  
3. Place dataset in `data/`  
4. Run notebook  

---

## Final Note

This project focuses on **system design, interpretability, and decision-making**, rather than maximizing model complexity. It demonstrates how classical NLP methods can be effectively used to build reliable AI-assisted workflows.
