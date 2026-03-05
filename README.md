<h1 align="center">Children's Speech Analysis Pipeline</h1>
<div align="center">
  
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Whisper](https://img.shields.io/badge/Whisper-ASR-412991?style=flat&logo=openai&logoColor=white)
![LoRA](https://img.shields.io/badge/LoRA-Fine--tuning-brightgreen?style=flat)
</div>

> An end-to-end pipeline for analyzing children's speech from mixed adult-child recordings, using a fine-tuned ASR model and NLP-based linguistic analysis.

---

## 📌 Overview

Standard Automatic Speech Recognition (ASR) models struggle with children's atypical and unclear pronunciation patterns. This project addresses that gap by building an audio processing pipeline with speaker assignment and linguistic analysis — and fine-tuning a Swedish Whisper model to better handle children's speech.

---

## 🔎 Features

- **ASR** — [kb-whisper-large](https://huggingface.co/KBLab/kb-whisper-large) for speech-to-text transcription
- **Speaker Assignment** — Logistic regression model to separate child and adult speech segments
- **Linguistic Analysis** — [Stanza](https://stanfordnlp.github.io/stanza/) and [spaCy](https://spacy.io/) for NLP-based lexical analysis
- **LoRA Fine-tuning** — Low-Rank Adaptation to fine-tune kb-whisper-large on limited children's speech data within Colab's memory constrains, specializing the model for children's speech

---

## 📁 Project Structure

```
├── data/
│   ├── train/          # Training data
│   └── test/           # Test data
├── models/             # Saved model weights (not tracked in git)
├── data_loader.py      # Data loading utilities
├── functions.py        # Helper functions
├── lr_train.py         # Logistic regression training
└── main.py             # Full pipeline: data loading, model inference, optional LR training
```

---

## ➿ Procedure

### 1. Prepare Data
Place data in `data/train/` and `data/test/`.

> **Note**: This data is used for the logistic regression speaker assignment model only — separate from the dataset used to fine-tune the Whisper ASR model.

Each dataset split contains:
- Multiple `.wav` audio files (mixed adult-child recordings)
- A `.csv` file with transcriptions in the format: `[filename], [transcribed text]`

### 2. Run
```bash
python main.py
```
`main.py` handles the full pipeline — loading data, loading models, and running inference. Logistic regression training can be enabled and configured via parameters inside `main.py`.

> The LoRA fine-tuned Whisper model was trained separately on Google Colab and is loaded from a private Hugging Face repository.

---

## 📃 Results

Reduced WER from **0.23 → 0.157** through data cleaning, text postprocessing ([jiwer](https://github.com/jitsi/jiwer)), and hyperparameter tuning.

- **Base model** — performs better on longer audio files
- **LoRA model** — higher accuracy on shorter audio files, but hallucinates more severely

---

> 🔒 **Note**: Data used in this project is proprietary and not publicly available.
