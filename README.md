# PLM-based Classification of Direct and Indirect Speech Acts

Sentence-level classification of direct and indirect speech acts  
in Korean spoken utterances using a pre-trained language model (PLM).

This project focuses on **designing and evaluating a PLM-based experiment**
rather than building a production-ready classification system.

---

## Project Overview

- Input: Korean spoken utterance (sentence-level)
- Task: Binary classification (direct / indirect speech act)
- Approach: Fine-tuning a pre-trained language model
- Emphasis:
  - experimental design
  - handling class imbalance
  - evaluation strategy and result interpretation

---

## Tech Stack

- **Language**: Python  
- **Deep Learning**: PyTorch  
- **PLM**: HuggingFace Transformers  
- **Data Processing**: pandas  
- **Evaluation**: scikit-learn  
- **Environment**: Google Colab

---

## What I Implemented

### Data Processing
- Loaded and cleaned labeled Korean spoken utterance data
- Mapped speech act labels for classification
- Split data into train / validation / test sets
  - Conditional stratified split applied to mitigate minority class loss

### Model Training
- Loaded a pre-trained language model for sequence classification
- Fine-tuned the model on the speech act classification task
- Applied **class-weighted loss** to address severe class imbalance
- Customized training behavior using PyTorch-based logic

### Evaluation
- Evaluated model performance using:
  - accuracy
  - **macro-F1**
  - class-specific recall
- Generated confusion matrix and classification report
- Inspected misclassified examples for qualitative analysis

### Reproducibility
- Fixed random seed (`seed = 42`)
- Saved trained model, tokenizer, and evaluation outputs
- Stored metrics and predictions for consistent result sharing

---

## Results Summary

- The model successfully identified some indirect speech acts,
  particularly interrogative forms.
- Performance varied significantly due to extreme class imbalance.
- Misclassification analysis revealed structural limitations
  of sentence-level input without discourse context.

---

## Notes

- This repository includes a Colab notebook with **pre-computed results**.
- Original spoken utterance data is not included.
- Re-running the notebook may produce slightly different metrics
  due to data sparsity and small test sets.
- - The dataset used in this project is an internally provided, research-only corpus
  originally collected for linguistic analysis.

---

## Purpose

This project serves as a portfolio example demonstrating:

- Practical use of PyTorch for model training and customization
- Fine-tuning of pre-trained language models for linguistic analysis
- Experimental design and evaluation under real-world data constraints
