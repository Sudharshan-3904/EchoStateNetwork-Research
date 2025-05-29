# Echo State Network Research Project

This project is a research and prototyping environment for experimenting with Echo State Networks (ESNs) for natural language processing, specifically text generation, token prediction, and chatbot functionalities. It includes training scripts, inference modules, scraping utilities, and legacy experimental code.

---

## Table of Contents

- [Objectives](#objectives)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Usage Guide](#usage-guide)
- [Dataset Information](#dataset-information)
- [Utilities](#utilities)
- [Research Findings](#research-findings)
- [Future Works](#future-works)
- [Contributions](#contributions)
- [Acknowledgments](#acknowledgments)

---

## Objectives

- To build different variant of ESN
- To apply ESN to varous Hardware ( CPU, GPU, NPU )
- To validate the use of ECN for a chatbot
- To explore other applications for ESN's

---

## Project Structure

The files have been organized into the following structure for clarity and maintainability:

```bash
.
├──src/
│   ├── inference/
│   │   ├── chat_with_combined_model.py
│   │   ├── chat_with_combined_model_gpu.py
│   │   └── interactive_chatbot.py
│   ├── models/
│   │   └── ESN_model.py
│   ├── training/
│   │   ├── train_next_token_gpu.py
│   │   └── train_with_webscraping_gpu.py
│   ├── utils/
│   │   └── combine_multiple_esn.py
│   └── webscraping/
│       ├── web_scrape.py
│       └── web_token_scraping.py
│
├── utilities/
│   ├── ESN_chat.py
│   ├── ScrapeTrain.py
│   ├── combining_pkls/
│   │   ├── comb_esn_pkl.py
│   │   ├── comb_vect_pkl.py
│   │   ├── combine_pkl.py
│   │   └── vect_combine_pkl.py
│   └── wikitrain/
│       ├── ESN_gpu_full.py
│       ├── ESN_train.py
│       ├── ESN_train_gpu.py
│       ├── ESN_train_npu.py
│       └── ESN_train_raw.txt
│
├── data/
│   ├── scraped_websites.txt
│   └── urls.csv
│
├── docs/
│   └── Extraceted Guidelines.md
│
├── logs/
│   ├── batch_progress.txt
│   ├── batch_times.csv
│   └── error_log.txt
│
└── requirements.txt
```

---

## Setup Instructions

- Requirements:

  - Python 3.10 or higher

- Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

- Optional for GPU training:

  - torch with CUDA
  - sentence-transformers

---

## Usage Guide

**Train ESN with Web Scraping:**

```bash
python src/training/train_with_webscraping_gpu.py
```

**Train ESN on Token Prediction:**

```bash
python src/training/train_next_token_gpu.py
```

**Run Chatbot with Combined Model (GPU):**

```bash
python src/inference/chat_with_combined_model_gpu.py
```

**Interactive Terminal Chatbot:**

```bash
python src/inference/interactive_chatbot.py
```

---

## Dataset Information

- `data/urls.csv`: List of URLs used for crawling and scraping.
- `data/scraped_websites.txt`: List of web pages crawled.

Web scraping scripts are located in `src/webscraping/`.

---

## Utilities

The `utilities/` directory contains basic and experimental scripts that are not actively maintained but may be helpful for reference or revival.s

- `esn_chat.py` — basic chatbot prototype.
- `wikitrain/` — initial ESN training tests on Wikipedia datasets.
- `combining_pkls/` — experiments for merging model weights and vector pickles.

---

## Research Findings

- An ESN can handle small to mid term dependencies
- An ESN cannot handle long term dependecies required for Chatot like applications
- A separate vectorizer is needed for NLP
- ESN's cannot handle large vocabularies
- It can be used to assist a chatbot for simple tasks based on small substring
- ESN properties have the following implications:

  | Property        | Purpose             | Description                                | Optimal value ( Experimental )    |
  | --------------- | ------------------- | ------------------------------------------ | --------------------------------- |
  | input_size      | Interface Parameter | To receive and feed to reservoir           | `Based on input`                  |
  | reservoir_size  | Model Definition    | Larger increases dependency lifetime       | 1000 ( mid term dependencies )    |
  | output_size     | Interface Parameter | To map the reservoir to outptu             | `Based on application`            |
  | spectral_radius | Fine Tuning         | How far spread out the model is            | 0.95                              |
  | sparsity        | Fine Tuning         | How common the connection b/w two nodes is | 0.1 ( 10% chance for connection ) |
  | input_scaling   | Fine Tuning         | Adjusts inputs to pass to reservoir        | 1.0                               |
  | leaking_rate    | Fine Tuning         | Impact network stability                   | 1.0                               |

---

## Future Works

Echo State Networks are best suited for other input types and thus have applicaitons in other fields. To further continue further, ESN can be applied in applicaitons and the fields mentioned below:

- Signal processign and signal based controlling
- Time series prediciton (Stock Price Prediciton, etc. )
- Potentially Speeck Recognition
- Financial data modelling

---

## Contributions

Any futher collaborative exploration and experimentation learning is appreciated. The main goal of this repo is to act as a base code for understanding Echo State Networks in a practical way by applying. Please feel free to fork this repository and experiment.  

---

## Acknowledgments

This Research was conducted as an passion based exploration into the workings and applications of ESN primarily in Chatbot like applications. The general idea for the model architecture was designed based on papers published on the topic. A few key papers are provided under `docs`.

---
