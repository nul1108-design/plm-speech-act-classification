# PLM-based Classification of Direct and Indirect Speech Acts

## Overview
This project explores sentence-level classification of direct and indirect speech acts
in Korean spoken utterances using a pre-trained language model (PLM).
The goal is methodological exploration rather than production-level automation.

## Technologies
- Python
- PyTorch
- HuggingFace Transformers
- pandas, scikit-learn

## What I Did
- Preprocessed and labeled Korean spoken utterance data
- Designed a train/validation/test split considering class imbalance
- Implemented class-weighted loss with PyTorch
- Fine-tuned a pre-trained language model for speech act classification
- Evaluated performance using macro-F1 and class-specific recall
- Analyzed misclassified examples
- Saved models and evaluation results for reproducibility (seed=42)

## Notes
This repository contains a Colab notebook with fixed experimental results.
Re-running the notebook may lead to slightly different metrics due to data sparsity.
